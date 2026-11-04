/* Copyright 2019 The IREE Authors
   Copyright 2023 OpenXLA Authors. All Rights Reserved.

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

// Implements logic for lowering StableHLO scatter ops to Linalg dialect.
// These patterns are separated out to their own file to save on the compilation
// times.

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {
namespace {
bool isAssignment(stablehlo::ScatterOp op) {
  // Return true if the scatter op is equivalent to an assignment.
  // This means that there is only one op in the body, and it is a ReturnOp.
  // E.g.,
  // update_function =
  // ^bb0(%arg0: T, %arg1: T):
  //   return %arg1 : T
  // })
  Region &region = op.getUpdateComputation();
  Block &block = region.front();
  bool oneOperation = block.begin() == --block.end();
  if (!oneOperation) {
    return false;
  }

  stablehlo::ReturnOp returnOp =
      dyn_cast<stablehlo::ReturnOp>(block.getTerminator());
  if (!returnOp) {
    return false;
  }

  return returnOp.getOperands().front() == block.getArgument(1);
}

bool singleFullSlices(stablehlo::ScatterOp op) {
  // Return true if the scatter op is inserting the whole update tensor into the
  // input tensor. This means that all dims that are not in the
  // update_window_dims are size 1.

  auto update = op.getUpdates().front();
  auto updateTy = dyn_cast<RankedTensorType>(update.getType());
  if (!updateTy || !updateTy.hasStaticShape()) {
    return false;  // Can't verify without static shape
  }

  auto scatterDimNumbers = op.getScatterDimensionNumbers();
  auto updateWindowDims = scatterDimNumbers.getUpdateWindowDims();

  llvm::SmallDenseSet<int64_t> windowDimsSet(updateWindowDims.begin(),
                                             updateWindowDims.end());

  auto shape = updateTy.getShape();
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (!windowDimsSet.contains(i)) {
      if (shape[i] != 1) {
        // Found a non-window dimension that is not size-1
        return false;
      }
    }
  }
  return true;
}

bool isInsertSliceScatter(stablehlo::ScatterOp op) {
  // Return true if the scatter op is equivalent to an insert_slice

  // Requirement 1: has exactly one input, one update and one result tensor
  if (op.getInputs().size() != 1 || op.getUpdates().size() != 1 ||
      op.getResults().size() != 1) {
    return false;
  }

  // Requirement 2: is assignment (see isAssignment)
  if (!isAssignment(op)) {
    return false;
  }

  // Requirement 3: no batching
  // input_batching_dims = []
  // scatter_indices_batching_dims = []
  auto scatterDimNumbers = op.getScatterDimensionNumbers();
  if (!scatterDimNumbers.getInputBatchingDims().empty()) {
    return false;
  }

  // Requirement 4: we are inserting the whole %update into a dimension of
  // %input
  if (!singleFullSlices(op)) {
    return false;
  }

  // Requirement 5: scatter indices is a static tensor of size 1
  auto indicesType = cast<RankedTensorType>(op.getScatterIndices().getType());
  if (!indicesType.hasStaticShape() || indicesType.getNumElements() != 1) {
    return false;
  }

  return true;
}

/// Pattern to lower relevant stablehlo::ScatterOps to tensor.insert_slice ops
struct ReduceOpToInsertSliceConverter final
    : public OpConversionPattern<stablehlo::ScatterOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isInsertSliceScatter(op)) {
      return failure();
    }

    auto input = op.getInputs().front();
    auto update = op.getUpdates().front();
    auto scatterIndices = op.getScatterIndices();

    auto inputTy = cast<RankedTensorType>(input.getType());
    auto updateTy = cast<RankedTensorType>(update.getType());
    auto inputShape = inputTy.getShape();
    auto updateShape = updateTy.getShape();

    auto scatterDimNumbers = op.getScatterDimensionNumbers();
    auto insertedWindowDims = scatterDimNumbers.getInsertedWindowDims();

    SmallVector<Value> dynOffsets, dynSizes, dynStrides;
    SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
    Location loc = op.getLoc();
    bool sameRank = inputTy.getRank() == updateTy.getRank();

    for (size_t i = 0, updateDim = 0; i < inputShape.size(); i++) {
      if (llvm::is_contained(insertedWindowDims, i)) {
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        int64_t rank = cast<ShapedType>(scatterIndices.getType()).getRank();
        SmallVector<Value> indices;
        for (int64_t i = 0; i < rank; ++i) {
          indices.push_back(zero);
        }
        auto extractOp =
            rewriter.create<tensor::ExtractOp>(loc, scatterIndices, indices);
        auto indexCastOp = rewriter
                               .create<arith::IndexCastOp>(
                                   loc, rewriter.getIndexType(), extractOp)
                               .getResult();

        // Offset is dynamic, based on the index we extract
        dynOffsets.push_back(indexCastOp);
        staticOffsets.push_back(ShapedType::kDynamic);
        staticSizes.push_back(1);
        if (sameRank) {
          if (updateShape[updateDim] != 1) {
            op->emitError(llvm::formatv("updateShape[{0}] must be 1, got {1}",
                                        updateDim, updateShape[updateDim]));
          }
          updateDim++;
        }

      } else {
        staticOffsets.push_back(0);
        staticSizes.push_back(updateShape[updateDim]);
        updateDim++;
      }
      staticStrides.push_back(1);
    }

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, update, input, dynOffsets, dynSizes, dynStrides, staticOffsets,
        staticSizes, staticStrides);
    return success();
  }
};
}  // namespace

namespace detail {
void populateStablehloScatterToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns, bool enablePrimitiveOps) {
  // Ensure specialized patterns are higher priority than their generic
  // versions.
  patterns->add<ReduceOpToInsertSliceConverter>(typeConverter, context,
                                                PatternBenefit(2));
}
}  // namespace detail
}  // namespace mlir::stablehlo
