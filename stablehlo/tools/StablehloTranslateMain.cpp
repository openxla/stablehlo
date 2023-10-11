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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/reference/InterpreterApi.h"
#include "stablehlo/reference/InterpreterOps.h"
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

TranslateFromMLIRRegistration interpretRegistration(
    "interpret", "Interpreter for StableHLO",
    [](ModuleOp module, raw_ostream &os) {
      stablehlo::DefaultInterpreterFallback fallback;
      stablehlo::InterpreterConfiguration config;
      config.probeInstrumentationDir = probeOutputDir.getValue();
      config.fallback = &fallback;
      config.stream = &os;

      auto status_or_results = runInterpreter(module, /*inputs=*/{}, config);

      if (status_or_results.getError()) {
        return success(false);
      }

      return success();
    },
    [](DialectRegistry &registry) {
      registry.insert<func::FuncDialect>();
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
