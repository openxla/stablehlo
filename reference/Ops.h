#ifndef STABLHLO_REFERENCE_OPS_H
#define STABLHLO_REFERENCE_OPS_H

#include "Tensor.h"
#include "dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

Tensor eval(AddOp op, const Tensor &lhs, const Tensor &rhs);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLHLO_REFERENCE_OPS_H
