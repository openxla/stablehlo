#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
