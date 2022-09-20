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
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

Tensor eval(AddOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());

  DimensionIterator dimIter(lhs.getType().getShape());
  while (dimIter.canIncrement()) {
    auto nextIndex = dimIter.getNextIndex();
    result.set(nextIndex, lhs.get(nextIndex) + rhs.get(nextIndex));
  }
  return result;
}

Tensor eval(CeilOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto i = 0; i < operand.getNumElements(); ++i) {
    result.set(i, ceil(operand.get(i)));
  }
  return result;
}

Tensor eval(ConstantOp op, ElementsAttr value) {
  return Tensor(value.cast<DenseElementsAttr>());
}

Tensor eval(CosineOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto i = 0; i < operand.getNumElements(); ++i) {
    result.set(i, cosine(operand.get(i)));
  }
  return result;
}

Tensor eval(FloorOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto i = 0; i < operand.getNumElements(); ++i) {
    result.set(i, floor(operand.get(i)));
  }
  return result;
}

Tensor eval(SineOp op, const Tensor &operand) {
  Tensor result(op.getType());

  DimensionIterator dimIter(operand.getType().getShape());
  while (dimIter.canIncrement()) {
    auto nextIndex = dimIter.getNextIndex();
    result.set(nextIndex, sine(operand.get(nextIndex)));
  }
  return result;
}

Tensor eval(TanhOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto i = 0; i < operand.getNumElements(); ++i) {
    result.set(i, tanh(operand.get(i)));
  }
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
