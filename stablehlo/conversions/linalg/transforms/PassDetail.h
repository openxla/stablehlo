#ifndef STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_PASSDETAIL_H
#define STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace stablehlo {

#define GEN_PASS_CLASSES
#include "stablehlo/conversions/linalg/transforms/Passes.h.inc"

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_PASSDETAIL_H
