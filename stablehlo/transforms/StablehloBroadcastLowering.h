/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_TRANSFORMS_OPBROADCASTUTILS_H_
#define STABLEHLO_TRANSFORMS_OPBROADCASTUTILS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace stablehlo {

///////
// Numpy broadcasting with support for bounded dynamism.

// Struct that represents a dim size of a tensor and possible dynamic value to
// match. If dimension is not dynamic, bound_op is set to std::nullopt. If
// dimension is bounded, the resulting dimension should be padded to `size` then
// marked dynamic using:
//   runtime_size = get_dimension_size(bound_op, dim=bound_op_dim)
//   T = set_dimension_size(T, dim=bound_op_dim, runtime_size)
//
struct DimensionInfo {
  int64_t size;
  std::optional<Value> boundOp = std::nullopt;
  int64_t boundOpDim = -1;
};

using Dimensions = SmallVector<DimensionInfo>;
std::string toString(const Dimensions& dims);

// Returns the dimensions of the given op, or failure if the op's type is not a
// ranked tensor.
FailureOr<Dimensions> getDimensions(Value op);

// Returns the ranked tensor type with the given dimensions and element type.
mlir::RankedTensorType getRankedTensorType(const Dimensions& dims,
                                           mlir::Type element_type);

// Returns the common shape these ops would broadcast to, or an error if the
// ops are not broadcastable.
FailureOr<Dimensions> getNumpyBroadcastShape(Location loc, ArrayRef<Value> ops);

// Apply numpy broadcasting to the given operands, returning an error if any
// operands are not broadcastable.
FailureOr<SmallVector<Value>> numpyBroadcastIfNeeded(OpBuilder& builder,
                                                     ArrayRef<Value> operands);

// Apply numpy broadcasting to the given operand, returning an error if the
// operand is not broadcastable.
FailureOr<Value> numpyBroadcastIfNeeded(OpBuilder& builder, Value input,
                                        const Dimensions& shape);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_OPBROADCASTUTILS_H_
