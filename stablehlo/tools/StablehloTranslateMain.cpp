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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/reference/InterpreterValue.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/tests/CheckOps.h"

namespace mlir {

namespace {

stablehlo::Tensor makeBooleanTensor(MLIRContext *context, bool value) {
  auto builder = Builder(context);
  auto type = RankedTensorType::get({}, builder.getI1Type());
  auto res = DenseElementsAttr::get(type, builder.getBoolAttr(true));
  return stablehlo::makeTensor(res);
}

llvm::Error wrapStatus(llvm::Error status, llvm::StringRef funcName,
                       llvm::StringRef fallbackName) {
  if (status)
    return stablehlo::invalidArgument(
        "Error evaluating function: %s. \n\tFallback for %s failed: %s",
        funcName.data(), fallbackName.data(),
        toString(std::move(status)).c_str());
  return llvm::Error::success();
}

llvm::Error evalCustomCallCheckEq(stablehlo::CustomCallOp op,
                                  stablehlo::Scope &scope) {
  if (op->getNumOperands() != 2)
    return stablehlo::invalidArgument("Unsupported op: %s",
                                      debugString(op).c_str());

  auto actualResult = scope.findTensors(op->getOperands())[0];
  auto expectedResult = scope.findTensors(op->getOperands())[1];
  bool isInt = expectedResult.getElementType().isa<IntegerType>();
  auto status =
      isInt ? stablehlo::check::evalExpectEqOp(actualResult, expectedResult)
            : stablehlo::check::evalExpectAlmostEqOp(actualResult,
                                                     expectedResult);
  if (status)
    scope.add(op.getResults(), stablehlo::InterpreterValue(
                                   makeBooleanTensor(op->getContext(), false)));
  else
    scope.add(op.getResults(), stablehlo::InterpreterValue(
                                   makeBooleanTensor(op->getContext(), true)));

  return status;
}

llvm::Error interpreterFallback(Operation &op, stablehlo::Process *process,
                                stablehlo::Scope &scope,
                                llvm::StringRef funcName) {
  if (auto customCall = dyn_cast<stablehlo::CustomCallOp>(op)) {
    if (customCall.getCallTargetName() == "check.eq") {
      auto status = evalCustomCallCheckEq(customCall, scope);
      return wrapStatus(std::move(status), funcName,
                        "stablehlo.custom_call(@check.eq)");
    }

    return stablehlo::invalidArgument("Unsupported custom call: %s",
                                      debugString(op).c_str());
  }

  if (auto expectAlmostEqOp =
          dyn_cast<stablehlo::check::ExpectAlmostEqOp>(op)) {
    auto runtimeLhs = scope.findTensor(expectAlmostEqOp.getLhs());
    auto runtimeRhs = scope.findTensor(expectAlmostEqOp.getRhs());
    auto status =
        stablehlo::check::evalExpectAlmostEqOp(runtimeLhs, runtimeRhs);
    return wrapStatus(std::move(status), funcName, "check.expect_almost_eq");
  }

  if (auto expectAlmostEqConstOp =
          dyn_cast<stablehlo::check::ExpectAlmostEqConstOp>(op)) {
    auto runtimeOperand = scope.findTensor(expectAlmostEqConstOp.getLhs());
    auto status = stablehlo::check::evalExpectAlmostEqConstOp(
        runtimeOperand, expectAlmostEqConstOp.getValue());
    return wrapStatus(std::move(status), funcName,
                      "check.expect_almost_eq_const");
  }

  if (auto expectEqOp = dyn_cast<stablehlo::check::ExpectEqOp>(op)) {
    auto runtimeLhs = scope.findTensor(expectEqOp.getLhs());
    auto runtimeRhs = scope.findTensor(expectEqOp.getRhs());
    auto status = stablehlo::check::evalExpectEqOp(runtimeLhs, runtimeRhs);
    return wrapStatus(std::move(status), funcName, "check.expect_eq");
  }

  if (auto expectEqConstOp = dyn_cast<stablehlo::check::ExpectEqConstOp>(op)) {
    auto runtimeOperand = scope.findTensor(expectEqConstOp.getLhs());
    auto status = stablehlo::check::evalExpectEqConstOp(
        runtimeOperand, expectEqConstOp.getValue());
    return wrapStatus(std::move(status), funcName, "check.expect_eq_const");
  }

  if (auto runParallelOp =
          dyn_cast<stablehlo::interpreter::RunParallelOp>(op)) {
    auto runtimeOperands = scope.find(runParallelOp.getInputs());
    std::queue<StringAttr> infeed;
    if (auto infeedAttr = runParallelOp.getInfeed())
      for (auto &value : infeedAttr->getValue())
        infeed.push(value.cast<FlatSymbolRefAttr>().getAttr());

    SmallVector<SmallVector<StringAttr>> programs(
        runParallelOp.getPrograms().size());
    for (auto [i, replica] : llvm::enumerate(runParallelOp.getPrograms()))
      for (auto &program : replica.cast<ArrayAttr>())
        programs[i].push_back(program.cast<FlatSymbolRefAttr>().getAttr());

    SymbolTable symbolTable{op.getParentOfType<ModuleOp>()};
    auto results = stablehlo::interpreter::evalRunParallelOp(
        runtimeOperands, infeed, programs, symbolTable);
    scope.add(runParallelOp.getResults(), results);
    return wrapStatus(llvm::Error::success(), funcName,
                      "interpreter.run_parallel");
  }

  return stablehlo::invalidArgument("Unsupported op: %s",
                                    debugString(op).c_str());
}

}  // namespace

TranslateFromMLIRRegistration interpretRegistration(
    "interpret", "Interpreter for StableHLO",
    [](ModuleOp module, raw_ostream &os) {
      auto numFuncs = 0;
      bool hasMain = false;
      module.walk([&](func::FuncOp funcOp) {
        if (funcOp.getSymName() == "main") hasMain = true;
        numFuncs++;
      });

      if (numFuncs > 1 && !hasMain)
        llvm::report_fatal_error(
            "Must have \"main\" function when multiple FuncOps are present");

      auto walkResult = module.walk([&](func::FuncOp funcOp) {
        if (numFuncs > 1 && funcOp.getSymName() != "main")
          return WalkResult::advance();

        auto interpreterFallbackFn =
            [&](Operation &op, stablehlo::Process *process,
                stablehlo::Scope &scope) -> llvm::Error {
          return interpreterFallback(op, process, scope, funcOp.getSymName());
        };

        // Run the test model.
        auto results =
            stablehlo::eval(funcOp.getBody(), /*args=*/{}, /*process=*/nullptr,
                            /*parent=*/nullptr, interpreterFallbackFn);

        // Dump the results.
        for (auto &result : results) result.print(os);
        return WalkResult::advance();
      });

      return success(!walkResult.wasInterrupted());
    },
    [](DialectRegistry &registry) {
      registry.insert<func::FuncDialect>();
      registry.insert<stablehlo::check::CheckDialect>();
      registry.insert<stablehlo::interpreter::InterpreterDialect>();
      registry.insert<stablehlo::StablehloDialect>();
    });

llvm::cl::opt<std::string> targetOption(
    "target", llvm::cl::desc("Target version for serialization"),
    llvm::cl::init(""));

llvm::cl::opt<bool> stripDebuginfoOption(
    "strip-debuginfo", llvm::cl::desc("Strip debug info from all operations"),
    llvm::cl::init(false));

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
