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

#include <cstdint>
#include "mlir/IR/AsmState.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/compatibility/DialectCompatibility.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace mlir {
namespace stablehlo {
void performRegistrations(MLIRContext * context) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  context->appendDialectRegistry(registry);

  // For `-emit-bytecode`
  registerAsmPrinterCLOptions();
}

static LogicalResult StablehloCompatibilitySerialization(
    llvm::SourceMgr &sourceMgr, llvm::raw_ostream &output, MLIRContext *context) {
  performRegistrations(context);
  OwningOpRef<Operation *> module =
      compatibility::parseWithCompat(sourceMgr, context);
  if (!module) {
    return failure();
  }

  // int64_t targetVersion = 20;  // FIXME
  bool emitBytecode = false;   // FIXME

  return compatibility::writeWithCompat(module.get(), {},
                                        emitBytecode, output);
}

}  // namespace stablehlo

TranslateRegistration stablehlo_compat(
    "compat", stablehlo::StablehloCompatibilitySerialization);

}  //  namespace mlir

int main(int argc, char **argv) {
  return failed(
      mlir::mlirTranslateMain(argc, argv, "StableHLO compatibility serializer\n"));
}
