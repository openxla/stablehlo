#include "Ops.h"
#include "Element.h"

namespace mlir {
namespace stablehlo {

Tensor eval(AddOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto i = 0; i < lhs.getNumElements(); ++i) {
    result.set(i, lhs.get(i) + rhs.get(i));
  }
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
