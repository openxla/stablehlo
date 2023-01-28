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

#ifndef STABLEHLO_REFERENCE_OPS_H
#define STABLEHLO_REFERENCE_OPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

Tensor eval_add(Type resultType, const Tensor &lhs, const Tensor &rhs);
Tensor eval_and(Type resultType, const Tensor &lhs, const Tensor &rhs);
Tensor eval_ceil(Type resultType, const Tensor &operand);
Tensor eval_constant(const ElementsAttr &value);
Tensor eval_cosine(Type resultType, const Tensor &operand);
Tensor eval_floor(Type resultType, const Tensor &operand);
Tensor eval_iota(Type resultType, uint64_t iotaDimension);
Tensor eval_max(Type resultType, const Tensor &lhs, const Tensor &rhs);
Tensor eval_min(Type resultType, const Tensor &lhs, const Tensor &rhs);
Tensor eval_multiply(Type resultType, const Tensor &lhs, const Tensor &rhs);
Tensor eval_neg(Type resultType, const Tensor &operand);
Tensor eval_not(Type resultType, const Tensor &operand);
Tensor eval_or(Type resultType, const Tensor &lhs, const Tensor &rhs);
Tensor eval_reshape(Type resultType, const Tensor &operand);
Tensor eval_sine(Type resultType, const Tensor &operand);
Tensor eval_subtract(Type resultType, const Tensor &lhs, const Tensor &rhs);
Tensor eval_tanh(Type resultType, const Tensor &operand);
Tensor eval_transpose(Type resultType, const Tensor &operand,
                      const DenseElementsAttr &permutation);
Tensor eval_xor(Type resultType, const Tensor &lhs, const Tensor &rhs);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_OPS_H
