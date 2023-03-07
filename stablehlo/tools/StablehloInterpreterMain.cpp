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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/tests/CheckOps.h"

namespace mlir {

TranslateFromMLIRRegistration interpretRegistration(
    "interpret", "Interpreter for StableHLO",
    [](ModuleOp module, raw_ostream &os) {
      auto walkResult = module.walk([&](func::FuncOp funcOp) {
        auto evalCheckOps = [&](Operation &op,
                                stablehlo::Scope &scope) -> llvm::Error {
          if (auto expectAlmostEqConstOp =
                  dyn_cast<stablehlo::check::ExpectAlmostEqConstOp>(op)) {
            stablehlo::Tensor runtimeOperand =
                scope.find(expectAlmostEqConstOp.getLhs());
            auto status = stablehlo::check::evalExpectAlmostEqConstOp(
                runtimeOperand, expectAlmostEqConstOp.getValue());
            if (status)
              return stablehlo::invalidArgument(
                  "Error evaluating function: %s. \n\tCheck "
                  "expect_almost_eq_const failed: "
                  "%s",
                  funcOp.getSymName().str().c_str(),
                  toString(std::move(status)).c_str());
          } else if (auto expectAlmostEqOp =
                         dyn_cast<stablehlo::check::ExpectAlmostEqOp>(op)) {
            stablehlo::Tensor runtimeLhs =
                scope.find(expectAlmostEqOp.getLhs());
            stablehlo::Tensor runtimeRhs =
                scope.find(expectAlmostEqOp.getRhs());
            auto status =
                stablehlo::check::evalExpectAlmostEqOp(runtimeLhs, runtimeRhs);
            if (status)
              return stablehlo::invalidArgument(
                  "Error evaluating function: %s. \n\tCheck expect_almost_eq "
                  "failed: "
                  "%s",
                  funcOp.getSymName().str().c_str(),
                  toString(std::move(status)).c_str());
          } else if (auto expectEqConstOp =
                         dyn_cast<stablehlo::check::ExpectEqConstOp>(op)) {
            stablehlo::Tensor runtimeOperand =
                scope.find(expectEqConstOp.getLhs());
            auto status = stablehlo::check::evalExpectEqConstOp(
                runtimeOperand, expectEqConstOp.getValue());
            if (status)
              return stablehlo::invalidArgument(
                  "Error evaluating function: %s. \n\tCheck expect_eq_const "
                  "failed: %s",
                  funcOp.getSymName().str().c_str(),
                  toString(std::move(status)).c_str());
          } else if (auto expectEqOp =
                         dyn_cast<stablehlo::check::ExpectEqOp>(op)) {
            stablehlo::Tensor runtimeLhs = scope.find(expectEqOp.getLhs());
            stablehlo::Tensor runtimeRhs = scope.find(expectEqOp.getRhs());
            auto status =
                stablehlo::check::evalExpectEqOp(runtimeLhs, runtimeRhs);
            if (status)
              return stablehlo::invalidArgument(
                  "Error evaluating function: %s. \n\tCheck expect_eq failed: "
                  "%s",
                  funcOp.getSymName().str().c_str(),
                  toString(std::move(status)).c_str());
          } else {
            return stablehlo::invalidArgument("Unsupported op: %s",
                                              debugString(op).c_str());
          }
          return llvm::Error::success();
        };

        // Run the test model.
        auto results = stablehlo::eval(funcOp.getBody(), {}, /*parent=*/nullptr,
                                       evalCheckOps);

        // Dump the results.
        for (auto &result : results) result.print(os);
        return WalkResult::advance();
      });

      return success(!walkResult.wasInterrupted());
    },
    [](DialectRegistry &registry) {
      registry.insert<func::FuncDialect>();
      registry.insert<stablehlo::check::CheckDialect>();
      registry.insert<stablehlo::StablehloDialect>();
    });

llvm::cl::opt<std::string> targetOption(
    "target", llvm::cl::desc("Target version for serialization"),
    llvm::cl::init(""));

TranslateFromMLIRRegistration serializeRegistration(
    "serialize", "Serialize StableHLO program into a portable artifact",
    [](ModuleOp module, raw_ostream &os) -> LogicalResult {
      return stablehlo::serializePortableArtifact(module, targetOption, os);
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
