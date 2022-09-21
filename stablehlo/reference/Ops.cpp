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
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, lhs.get(*it) + rhs.get(*it));
  }
  return result;
}

Tensor eval(CeilOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, ceil(operand.get(*it)));
  }
  return result;
}

Tensor eval(ConstantOp op, ElementsAttr value) {
  return Tensor(value.cast<DenseElementsAttr>());
}

Tensor eval(CosineOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, cosine(operand.get(*it)));
  }
  return result;
}

Tensor eval(FloorOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, floor(operand.get(*it)));
  }
  return result;
}

Tensor eval(NegOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, -operand.get(*it));
  }
  return result;
}

Tensor eval(ReshapeOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto operandIt = operand.index_begin(), resultIt = result.index_begin();
       operandIt != operand.index_end(); ++operandIt, ++resultIt) {
    result.set(*resultIt, operand.get(*operandIt));
  }
  return result;
}

Tensor eval(SineOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, sine(operand.get(*it)));
  }
  return result;
}

Tensor eval(SubtractOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, lhs.get(*it) - rhs.get(*it));
  }
  return result;
}

Tensor eval(TanhOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, tanh(operand.get(*it)));
  }
  return result;
}

namespace {

// Appies permutation `perm` to an array `array` where perm[i] indicates the
// location where the current array[i] goes.
void applyInPlacePermutation(std::vector<int64_t> &array,
                             const std::vector<int64_t> &perm) {
  size_t swapIdx;
  for (size_t i = 0; i < perm.size(); i++) {
    swapIdx = perm[i];
    while (swapIdx < i) {
      swapIdx = perm[swapIdx];
    }
    std::swap(array[i], array[swapIdx]);
  }
}

}  // namespace

Tensor eval(TransposeOp op, const Tensor &operand, DenseIntElementsAttr perm) {
  Tensor result(operand);
  result.setType(op.getType());

  // The operation itself does not require any copying but involves swapping
  // strides.
  std::vector<int64_t> stride(operand.getStrides());
  std::vector<int64_t> permutation(perm.getValues<int64_t>().begin(),
                                   perm.getValues<int64_t>().end());
  applyInPlacePermutation(stride, permutation);
  result.setStrides(stride);
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
