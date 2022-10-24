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

#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/compatibility/StablehloDialectCompatibility.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/tests/TestStablehloDialectCompatibility.h"

using namespace mlir;
using namespace mlir::stablehlo;

void performDialectRegistrations(MLIRContext *context) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  context->appendDialectRegistry(registry);
}

static LogicalResult serializeWithCompatibilityTest(llvm::SourceMgr &sourceMgr,
                                                    llvm::raw_ostream &output,
                                                    MLIRContext *context,
                                                    CompatOptions opts) {
  // Uses internal APIs to use test interface
  TestStablehloCompatibilityConverter interface(context);
  OwningOpRef<Operation *> module =
      stablehlo::detail::parseWithCompatImpl(sourceMgr, context, interface);
  if (!module) return failure();
  return stablehlo::detail::writeWithCompatImpl(module.get(), opts, output,
                                                interface);
}

static LogicalResult serializeWithCompatibilityMain(llvm::SourceMgr &sourceMgr,
                                                    llvm::raw_ostream &output,
                                                    MLIRContext *context,
                                                    CompatOptions opts) {
  // Use StablehloDialectCompatibility APIs
  OwningOpRef<Operation *> module = parseWithCompat(sourceMgr, context);
  if (!module) return failure();
  return writeWithCompat(module.get(), context, opts, output);
}

int main(int argc, char **argv) {
  // FIXME: This is how other tools implement arguments by the look of it, is
  // this correct? Couldn't find much about "translation registration options".
  static llvm::cl::opt<int64_t> targetVersion(
      "target",
      llvm::cl::desc(
          "Target verison for output (default to minimum supported)"),
      llvm::cl::init(-1));

  static llvm::cl::opt<bool> emitAssembly(
      "emit-assembly",
      llvm::cl::desc("Emit textual assembly format (default emits bytecode)"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> useTestConverter(
      "use-test-converter",
      llvm::cl::desc(
          "Use the test converter (for unit testing conversion machinery)"),
      llvm::cl::init(false));

  // Allow printer flags like `--mlir-print-op-generic`
  registerAsmPrinterCLOptions();

  TranslateRegistration stablehlo_compat(
      "compat", "StableHLO compatibility tool.",
      [&](llvm::SourceMgr &sourceMgr, llvm::raw_ostream &output,
          MLIRContext *context) {
        CompatOptions opts{targetVersion, emitAssembly};
        performDialectRegistrations(context);
        if (useTestConverter) {
          return serializeWithCompatibilityTest(sourceMgr, output, context,
                                                opts);
        }
        return serializeWithCompatibilityMain(sourceMgr, output, context, opts);
      });

  return failed(mlir::mlirTranslateMain(
      argc, argv, "StableHLO compatibility serializer\n"));
}
