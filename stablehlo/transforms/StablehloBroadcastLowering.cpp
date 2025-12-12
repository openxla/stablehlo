/* Copyright 2025 The StableHLO Authors.

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

#include "stablehlo/transforms/StablehloBroadcastLowering.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "stablehlo-broadcast-lowering"

namespace mlir {
namespace stablehlo {

/////
// Bounded dynamism broadcasting

namespace {

DimensionInfo getDimensionInfo(Value op, mlir::RankedTensorType tensorType,
                               TypeExtensionsAttr encoding, int64_t dim) {
  if (!encoding || !mlir::ShapedType::isDynamic(tensorType.getDimSize(dim)))
    return DimensionInfo{tensorType.getDimSize(dim)};

  return DimensionInfo{
      encoding.getBounds()[dim],
      op,
      dim,
  };
}

FailureOr<Dimensions> getNumpyBroadcastShapeWithBounds(Value op,
                                                       const Dimensions& a,
                                                       const Dimensions& b) {
  LLVM_DEBUG(llvm::dbgs() << "[getNumpyBroadcastShapeWithBounds] inputs: "
                          << toString(a) << " * " << toString(b) << "\n");
  size_t max_rank = std::max(a.size(), b.size());
  Dimensions result(max_rank);

  // Iterate from right to left (NumPy-style broadcasting)
  for (size_t i = 1; i <= max_rank; ++i) {
    size_t a_idx = a.size() - i;
    size_t b_idx = b.size() - i;
    size_t res_idx = max_rank - i;

    // Get DimensionInfo for the current index, padding with size 1 if out of
    // bounds.
    DimensionInfo dim_a =
        (a_idx >= 0 && a_idx < a.size()) ? a[a_idx] : DimensionInfo{1};
    DimensionInfo dim_b =
        (b_idx >= 0 && b_idx < b.size()) ? b[b_idx] : DimensionInfo{1};

    // Short circuit on size 1 dimensions.
    if (dim_a.size == 1) {
      result[res_idx] = dim_b;
      continue;
    }
    if (dim_b.size == 1) {
      result[res_idx] = dim_a;
      continue;
    }

    // If both LHS and RHS are not 1, dim size must match.
    if (dim_a.size != dim_b.size) {
      // FIXME
      return emitError(op.getLoc(), "incompatible shapes for broadcasting ")
             << dim_a.size << " and " << dim_b.size;
    }

    // If bounded both must be bounded
    if (dim_a.boundOp.has_value() != dim_b.boundOp.has_value()) {
      return emitError(op.getLoc(),
                       "cannot mix bounded and static dimensions in broadcast");
    }

    // LHS and RHS match, populate with one of the dimensions.
    result[res_idx] = dim_a;
  }

  LLVM_DEBUG(llvm::dbgs() << "[getNumpyBroadcastShapeWithBounds] result: "
                          << toString(result) << "\n");
  return result;
}

}  // namespace

FailureOr<Dimensions> getDimensions(Value op) {
  // Get tensor type
  mlir::RankedTensorType tensor_type = dyn_cast<RankedTensorType>(op.getType());
  if (!tensor_type)
    return emitError(op.getLoc(),
                     "expected ranked tensor type for broadcast inputs");

  auto encoding =
      mlir::dyn_cast_if_present<mlir::stablehlo::TypeExtensionsAttr>(
          tensor_type.getEncoding());

  Dimensions dimensions;
  dimensions.reserve(tensor_type.getRank());
  for (int64_t idx = 0; idx < tensor_type.getRank(); ++idx) {
    auto dimInfo = getDimensionInfo(op, tensor_type, encoding, idx);
    dimensions.push_back(dimInfo);
  }
  return dimensions;
}

mlir::RankedTensorType getRankedTensorType(const Dimensions& dims,
                                           mlir::Type element_type) {
  mlir::SmallVector<int64_t> shape;
  mlir::SmallVector<int64_t> bounds;
  shape.reserve(dims.size());
  for (const DimensionInfo& dim : dims) {
    if (dim.boundOp.has_value()) {
      shape.push_back(mlir::ShapedType::kDynamic);
      bounds.push_back(dim.size);
    } else {
      shape.push_back(dim.size);
      bounds.push_back(mlir::ShapedType::kDynamic);
    }
  }
  mlir::stablehlo::TypeExtensionsAttr encoding;
  if (!llvm::all_of(
          bounds, [](int64_t b) { return b == mlir::ShapedType::kDynamic; })) {
    encoding = mlir::stablehlo::TypeExtensionsAttr::get(
        element_type.getContext(), bounds);
  }
  return mlir::RankedTensorType::get(shape, element_type, encoding);
}

FailureOr<Dimensions> getNumpyBroadcastShape(OpBuilder& builder,
                                             ArrayRef<Value> ops) {
  if (ops.empty())
    return emitError(builder.getInsertionPoint()->getLoc(),
                     "requires at least one operand to broadcast");

  Value first = ops[0];
  auto bcastShapeOrFail = getDimensions(first);
  if (failed(bcastShapeOrFail)) return failure();
  Dimensions bcastShape = std::move(*bcastShapeOrFail);

  for (size_t i = 1; i < ops.size(); ++i) {
    Value currOp = ops[i];
    auto dims = getDimensions(currOp);
    if (failed(dims)) return failure();
    auto currBcastShapeOrFail =
        getNumpyBroadcastShapeWithBounds(currOp, bcastShape, *dims);
    if (failed(currBcastShapeOrFail)) return failure();
    bcastShape = std::move(*currBcastShapeOrFail);
  }
  return std::move(bcastShape);
}

std::string toString(const Dimensions& dims) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "tensor<";
  llvm::interleave(
      dims, os,
      [&](const DimensionInfo& dim) {
        os << (dim.boundOp.has_value() ? "b" : "") << dim.size;
      },
      "x");
  os << ">";
  return result;
}

FailureOr<SmallVector<Value>> numpyBroadcastIfNeeded(OpBuilder& builder,
                                                     ArrayRef<Value> operands) {
  // Figure out the broadcast shape
  auto bcastShapeOrFail = getNumpyBroadcastShape(builder, operands);
  if (failed(bcastShapeOrFail)) return failure();
  Dimensions bcastShape = std::move(*bcastShapeOrFail);

  // Apply to all operands
  SmallVector<Value> broadcastedOperands;
  for (auto operand : operands) {
    auto bcastOperand = numpyBroadcastIfNeeded(builder, operand, bcastShape);
    if (failed(bcastOperand)) return failure();
    broadcastedOperands.push_back(*bcastOperand);
  }
  return std::move(broadcastedOperands);
}

FailureOr<Value> numpyBroadcastIfNeeded(OpBuilder& builder, Value input,
                                        const Dimensions& shape) {
  LLVM_DEBUG(llvm::dbgs() << "[numpyBroadcastIfNeeded] Broadcasting input "
                          << input.getType() << " => " << toString(shape)
                          << "\n");
  auto loc = input.getLoc();
  mlir::RankedTensorType inputType =
      dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return emitError(loc, "expected ranked tensor type for broadcast inputs");
  mlir::RankedTensorType outputType =
      getRankedTensorType(shape, inputType.getElementType());

  // Short circuit if no broadcasting is needed.
  if (inputType == outputType) return input;

  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();
  if (inputRank > outputRank)
    return emitError(loc, "input rank must be <= output rank, got ")
           << inputRank << " vs " << outputRank;

  size_t rankDiff = outputRank - inputRank;
  auto inputShapeOrFail = getDimensions(input);
  if (failed(inputShapeOrFail)) return failure();
  Dimensions inputShape = std::move(*inputShapeOrFail);

  // Construct broadcast dimensions.
  auto broadcastDimensions =
      llvm::to_vector(llvm::seq<int64_t>(outputRank - inputRank, outputRank));

  // Construct the result type of the broadcast
  //  - If input is static and target shape is static, use static shape.
  //  - If input has bounded dim, target shape must be bounded, use bounded dim.
  //  - If input is not bounded, but target shape is bounded, broadcast to
  //    the padded shape then call SetDimensionSize to make dynamic.
  auto bcastShape = shape;
  for (int64_t i = 0; i < inputRank; ++i) {
    int64_t inputDimSize = inputShape[i].size;
    int64_t resultIdx = i + rankDiff;
    int64_t resultDimSize = shape[resultIdx].size;
    if (inputDimSize != 1 && inputDimSize != resultDimSize)
      return emitError(loc, "Cannot broadcast input: ")
             << inputType << " to target shape " << toString(shape);

    if (!inputShape[i].boundOp.has_value() &&
        shape[resultIdx].boundOp.has_value()) {
      // Use padded shape in broadcast.
      bcastShape[resultIdx] = DimensionInfo{shape[resultIdx].size};
    }
  }

  // Broadcast to padded size for remaining dimensions.
  for (size_t i = 0; i < rankDiff; ++i) {
    bcastShape[i] = DimensionInfo{shape[i].size};
  }

  // Insert broadcast ops
  mlir::RankedTensorType bcastType =
      getRankedTensorType(bcastShape, inputType.getElementType());
  LLVM_DEBUG(
      llvm::dbgs() << "[numpyBroadcastIfNeeded] Broadcast to padded type "
                   << bcastType << "\n");
  Value bcastOp = stablehlo::BroadcastInDimOp::create(
      builder, loc, bcastType, input, broadcastDimensions);
  if (bcastOp.getType() == outputType) return bcastOp;

  // Mark the padded broadcast as dynamic where the result is bounded.
  // Inserts `GetDimSize(boundOp)->SetDimSize(inputBcast)` for any bounded
  // dimensions that required broadcasting.
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!bcastShape[i].boundOp.has_value() && shape[i].boundOp.has_value()) {
      Value boundOp = shape[i].boundOp.value();
      auto dimSize = stablehlo::GetDimensionSizeOp::create(
          builder, loc, boundOp, shape[i].boundOpDim);
      bcastOp = stablehlo::SetDimensionSizeOp::create(builder, loc, bcastOp,
                                                      dimSize, i);
    }
  }
  return bcastOp;
}

}  // namespace stablehlo
}  // namespace mlir
