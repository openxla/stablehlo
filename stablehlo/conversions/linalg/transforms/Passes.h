#ifndef STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_PASSES_H
#define STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace stablehlo {

#include "stablehlo/conversions/linalg/transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createStablehloLegalizeToLinalgPass();

void registerStablehloLegalizeToLinalgPass();

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_PASSES_H
