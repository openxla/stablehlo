/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Value.h"
#include "stablehlo/tests/CheckOps.h"

namespace mlir {

llvm::cl::opt<std::string> probeOutputDir(
    "probe-output-dir",
    llvm::cl::desc("Directory for storing instrumented tensor values"),
    llvm::cl::init(""));

llvm::cl::opt<bool> stripDebuginfoOption(
    "strip-debuginfo", llvm::cl::desc("Strip debug info from all operations"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> targetOption(
    "target", llvm::cl::desc("Target version for serialization"),
    llvm::cl::init(""));

namespace {

stablehlo::Tensor makeBooleanTensor(MLIRContext *context, bool value) {
  auto builder = Builder(context);
  auto type = RankedTensorType::get({}, builder.getI1Type());
  auto res = DenseElementsAttr::get(type, builder.getBoolAttr(true));
  return stablehlo::makeTensor(res);
}

llvm::Error evalCustomCallCheckEq(stablehlo::CustomCallOp op,
                                  stablehlo::Scope &scope) {
  if (op->getNumOperands() != 2)
    return stablehlo::invalidArgument("Unsupported op: %s",
                                      debugString(op).c_str());

  auto actualResult = scope.findTensors(op->getOperands())[0];
  auto expectedResult = scope.findTensors(op->getOperands())[1];
  bool isInt = isa<IntegerType>(expectedResult.getElementType());
  auto status =
      isInt ? stablehlo::check::evalExpectEqOp(actualResult, expectedResult)
            : stablehlo::check::evalExpectAlmostEqOp(
                  actualResult, expectedResult, APFloat(0.0001));
  if (status)
    scope.add(op.getResults(), stablehlo::InterpreterValue(
                                   makeBooleanTensor(op->getContext(), false)));
  else
    scope.add(op.getResults(), stablehlo::InterpreterValue(
                                   makeBooleanTensor(op->getContext(), true)));

  return status;
}

/// The default fallback callback used by StableHLO for interpreter validation
/// and module instrumentation.
class StablehloTranslateInterpreterFallback
    : public stablehlo::InterpreterFallback {
 public:
  StablehloTranslateInterpreterFallback(
      const std::string &probeInstrumentationDir)
      : probeInstrumentationDir(probeInstrumentationDir) {}
  virtual llvm::Error operator()(Operation &op, stablehlo::Scope &scope,
                                 stablehlo::Process *process) final {
    llvm::StringRef funcName = op.getParentOfType<func::FuncOp>().getSymName();
    if (auto customCall = dyn_cast<stablehlo::CustomCallOp>(op)) {
      auto callTarget = customCall.getCallTargetName();
      if (callTarget == "check.eq") {
        auto status = evalCustomCallCheckEq(customCall, scope);
        return stablehlo::wrapFallbackStatus(
            std::move(status), funcName, "stablehlo.custom_call(@check.eq)");
      }
      if (customCall->getNumOperands() == 2) {
        // TODO: Replace with custom_call --> check dialect pass
        auto runtimeLhs = scope.findTensor(customCall->getOperand(0));
        auto runtimeRhs = scope.findTensor(customCall->getOperand(1));
        // TODO: Make tolerance more strict
        if (callTarget == "check.expect_close") {
          auto status =
              stablehlo::check::evalExpectCloseOp(runtimeLhs, runtimeRhs, 0, 3);
          return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                               "check.expect_close");
        }
        if (customCall.getCallTargetName() == "check.expect_eq") {
          auto status =
              stablehlo::check::evalExpectEqOp(runtimeLhs, runtimeRhs);
          return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                               "check.expect_eq");
        }
        if (customCall.getCallTargetName() == "check.expect_almost_eq") {
          auto status = stablehlo::check::evalExpectAlmostEqOp(
              runtimeLhs, runtimeRhs, APFloat(0.001));
          return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                               "check.expect_almost_eq");
        }
      }

      return stablehlo::invalidArgument("Unsupported custom call: %s",
                                        debugString(op).c_str());
    }

    if (auto expectAlmostEqOp =
            dyn_cast<stablehlo::check::ExpectAlmostEqOp>(op)) {
      auto runtimeLhs = scope.findTensor(expectAlmostEqOp.getLhs());
      auto runtimeRhs = scope.findTensor(expectAlmostEqOp.getRhs());
      auto tolerance = expectAlmostEqOp.getTolerance();
      auto status = stablehlo::check::evalExpectAlmostEqOp(
          runtimeLhs, runtimeRhs, tolerance);
      return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                           "check.expect_almost_eq");
    }

    if (auto expectAlmostEqConstOp =
            dyn_cast<stablehlo::check::ExpectAlmostEqConstOp>(op)) {
      auto runtimeOperand = scope.findTensor(expectAlmostEqConstOp.getLhs());
      auto tolerance = expectAlmostEqConstOp.getTolerance();
      auto status = stablehlo::check::evalExpectAlmostEqConstOp(
          runtimeOperand, expectAlmostEqConstOp.getValue(), tolerance);
      return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                           "check.expect_almost_eq_const");
    }

    if (auto expectEqOp = dyn_cast<stablehlo::check::ExpectEqOp>(op)) {
      auto runtimeLhs = scope.findTensor(expectEqOp.getLhs());
      auto runtimeRhs = scope.findTensor(expectEqOp.getRhs());
      auto status = stablehlo::check::evalExpectEqOp(runtimeLhs, runtimeRhs);
      return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                           "check.expect_eq");
    }

    if (auto expectEqConstOp =
            dyn_cast<stablehlo::check::ExpectEqConstOp>(op)) {
      auto runtimeOperand = scope.findTensor(expectEqConstOp.getLhs());
      auto status = stablehlo::check::evalExpectEqConstOp(
          runtimeOperand, expectEqConstOp.getValue());
      return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                           "check.expect_eq_const");
    }

    if (auto expectCloseOp = dyn_cast<stablehlo::check::ExpectCloseOp>(op)) {
      auto runtimeActual = scope.findTensor(expectCloseOp.getActual());
      auto runtimeExpected = scope.findTensor(expectCloseOp.getExpected());
      auto status = stablehlo::check::evalExpectCloseOp(
          runtimeActual, runtimeExpected, expectCloseOp.getMinUlpDifference(),
          expectCloseOp.getMaxUlpDifference());
      return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                           "check.expect_close");
    }

    if (auto expectSerializedEqOp =
            dyn_cast<stablehlo::check::ExpectSerializedEqOp>(op)) {
      auto runtimeOperand =
          scope.findTensor(expectSerializedEqOp.getExpected());
      auto status = stablehlo::check::evalExpectSerializedEqOp(
          runtimeOperand, expectSerializedEqOp.getProbeId(),
          probeInstrumentationDir, expectSerializedEqOp.getIteration());
      return stablehlo::wrapFallbackStatus(std::move(status), funcName,
                                           "check.expect_serialized_eq");
    }

    return stablehlo::invalidArgument("Unsupported op: %s",
                                      debugString(op).c_str());
  }

 private:
  // The directory in which tensors instrumented by way of `interpreter.probe`
  // will have their data serialized to.
  const std::string probeInstrumentationDir;
};

}  // namespace

TranslateFromMLIRRegistration interpretRegistration(
    "interpret", "Interpreter for StableHLO",
    [](ModuleOp module, raw_ostream &os) -> LogicalResult {
      stablehlo::InterpreterConfiguration config;
      config.probeInstrumentationDir = probeOutputDir.getValue();
      config.fallback = std::make_unique<StablehloTranslateInterpreterFallback>(
          config.probeInstrumentationDir);

      llvm::SmallVector<stablehlo::InterpreterValue> inputs;
      auto results = evalModule(module, inputs, config);
      if (failed(results)) return failure();

      for (auto &result : *results) {
        result.print(os);
        os << '\n';
      }

      return success();
    },
    [](DialectRegistry &registry) {
      registry.insert<func::FuncDialect>();
      registry.insert<quant::QuantDialect>();
      registry.insert<stablehlo::check::CheckDialect>();
      registry.insert<stablehlo::interpreter::InterpreterDialect>();
      registry.insert<stablehlo::StablehloDialect>();
    });

TranslateFromMLIRRegistration serializeRegistration(
    "serialize", "Serialize StableHLO program into a portable artifact",
    [](ModuleOp module, raw_ostream &os) -> LogicalResult {
      std::string targetVersion = targetOption.getValue();
      if (targetVersion == "current")
        targetVersion = vhlo::Version::getCurrentVersion().toString();

      if (stripDebuginfoOption) {
        PassManager pm(module->getContext());
        pm.addPass(createStripDebugInfoPass());
        if (failed(pm.run(module)))
          return module.emitError("failed to strip debuginfo");
      }

      return stablehlo::serializePortableArtifact(module, targetVersion, os);
    },
    [](DialectRegistry &registry) {
      mlir::registerAllDialects(registry);
      mlir::stablehlo::registerAllDialects(registry);
    });

TranslateToMLIRRegistration deserializeRegistration(
    "deserialize", "Deserialize a portable artifact into a StableHLO program",
    [](llvm::StringRef input, mlir::MLIRContext *context) {
      return stablehlo::deserializePortableArtifact(input, context);
    },
    [](DialectRegistry &registry) {
      mlir::registerAllDialects(registry);
      mlir::stablehlo::registerAllDialects(registry);
    });

}  //  namespace mlir

int main(int argc, char **argv) {
  return failed(
      mlir::mlirTranslateMain(argc, argv, "StableHLO interpreter driver\n"));
}
