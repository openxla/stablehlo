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

#include "stablehlo/reference/Ops.h"

#include "stablehlo/reference/Element.h"

namespace mlir {
namespace stablehlo {

Tensor eval(AddOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto i = 0; i < lhs.getNumElements(); ++i) {
    result.set(i, lhs.get(i) + rhs.get(i));
  }
  return result;
}

Tensor eval(ConstantOp op, ElementsAttr value) {
  return Tensor(value.cast<DenseElementsAttr>());
}

}  // namespace stablehlo
}  // namespace mlir
