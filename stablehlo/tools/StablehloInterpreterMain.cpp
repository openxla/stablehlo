/* Copyright 2022 The StableHLO Authors.

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

#include "mlir/IR/OpDefinition.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Interpreter.h"

namespace mlir {

TranslateFromMLIRRegistration stablehlo_interpreter(
    "interpret", "Interpreter for StableHLO",
    [](ModuleOp module, raw_ostream &os) {
      auto walkResult = module.walk([&](func::FuncOp funcOp) {
        os << "\nEvaluated results of function: " << funcOp.getSymName()
           << "\n";

        // Run the test model.
        auto results = mlir::stablehlo::eval(funcOp, {});
        if (!results) {
          llvm::errs() << toString(results.takeError());
          return WalkResult::interrupt();
        }

        // Dump the results.
        for (auto &result : *results) result.print(os);
        return WalkResult::advance();
      });

      return success(!walkResult.wasInterrupted());
    },
    [](DialectRegistry &registry) {
      registry.insert<func::FuncDialect>();
      registry.insert<stablehlo::StablehloDialect>();
    });

}  //  namespace mlir

int main(int argc, char **argv) {
  return failed(
      mlir::mlirTranslateMain(argc, argv, "StableHLO interpreter driver\n"));
}
