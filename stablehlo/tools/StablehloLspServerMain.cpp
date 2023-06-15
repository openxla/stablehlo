#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
