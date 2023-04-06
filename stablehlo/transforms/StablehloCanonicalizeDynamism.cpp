/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZEDYNAMISMPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

struct CanonicalizeDynamicBroadcastInDimOpPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern ignores and discards the output_dimensions operand as well
    // as the known_expanding_dimensions and known_nonexpanding_dimensions
    // attributes. We rely on the verifier to make sure that their values are
    // consistent with the result type.
    if (!op.getOperand().getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static operand type");
    if (!op.getType().cast<ShapedType>().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op.getType(), op.getOperand(), op.getBroadcastDimensions());
    return success();
  }
};

struct CanonicalizeDynamicConvOpPattern
    : public OpRewritePattern<DynamicConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicConvOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> padding;
    if (!succeeded(hlo::matchInts(op.getDPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected static padding");
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(padding.size()) / 2, 2},
                              rewriter.getI64Type()),
        padding);
    rewriter.replaceOpWithNewOp<ConvolutionOp>(
        op, op.getType(), op.getLhs(), op.getRhs(), op.getWindowStridesAttr(),
        paddingAttr, op.getLhsDilationAttr(), op.getRhsDilationAttr(),
        op.getWindowReversalAttr(), op.getDimensionNumbers(),
        op.getFeatureGroupCount(), op.getBatchGroupCount(),
        op.getPrecisionConfigAttr());
    return success();
  }
};

struct CanonicalizeDynamicGatherOpPattern
    : public OpRewritePattern<DynamicGatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicGatherOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> sliceSizes;
    if (!succeeded(hlo::matchInts(op.getSliceSizes(), sliceSizes)))
      return rewriter.notifyMatchFailure(op, "expected static slice_sizes");
    rewriter.replaceOpWithNewOp<GatherOp>(
        op, op.getOperand(), op.getStartIndices(), op.getDimensionNumbersAttr(),
        rewriter.getI64TensorAttr(sliceSizes), op.getIndicesAreSortedAttr());
    return success();
  }
};
struct CanonicalizeDynamicIotaOpPattern
    : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicIotaOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern ignores and discards the output_shape operand. We rely on
    // the verifier to make sure that its value is consistent with result type.
    if (!op.getType().cast<ShapedType>().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<IotaOp>(op, op.getType(),
                                        op.getIotaDimension());
    return success();
  }
};

struct CanonicalizeDynamicPadOpPattern : public OpRewritePattern<DynamicPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicPadOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> edgePaddingLow, edgePaddingHigh, interiorPadding;
    if (!succeeded(hlo::matchInts(op.getEdgePaddingLow(), edgePaddingLow)))
      return rewriter.notifyMatchFailure(op, "expected static low");
    if (!succeeded(hlo::matchInts(op.getEdgePaddingHigh(), edgePaddingHigh)))
      return rewriter.notifyMatchFailure(op, "expected static high");
    if (!succeeded(hlo::matchInts(op.getInteriorPadding(), interiorPadding)))
      return rewriter.notifyMatchFailure(op, "expected static interior");
    rewriter.replaceOpWithNewOp<PadOp>(
        op, op.getOperand(), op.getPaddingValue(),
        rewriter.getI64TensorAttr(edgePaddingLow),
        rewriter.getI64TensorAttr(edgePaddingHigh),
        rewriter.getI64TensorAttr(interiorPadding));
    return success();
  }
};

struct CanonicalizeDynamicReshapeOpPattern
    : public OpRewritePattern<DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern ignores and discards the output_shape operand. We rely on
    // the verifier to make sure that its value is consistent with result type.
    if (!op.getType().cast<ShapedType>().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
    return success();
  }
};

struct CanonicalizeRealDynamicSliceOpToDynamicSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    // This rewrite only works for unit strides because DynamicSliceOp
    // doesn't support strides (i.e. it implicitly has unit strides).
    SmallVector<int64_t> strides;
    if (!succeeded(hlo::matchInts(op.getStrides(), strides)))
      return rewriter.notifyMatchFailure(op, "expected static strides");
    if (!llvm::all_of(strides, [&](int64_t stride) { return stride == 1; }))
      return rewriter.notifyMatchFailure(op, "expected unit strides");

    // Check that slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    if (!matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) &&
        !matchPattern(op.getLimitIndices(),
                      m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices)))
      return rewriter.notifyMatchFailure(
          op, "expected limit indices equal to start indices plus constant");

    // RealDynamicSliceOp can take tensors of integer or index element types.
    // DynamicSliceOp::slice_sizes only supports i64 element type.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<int64_t> sliceSizes;
    for (auto element : sliceSizesAttr.getValues<APInt>()) {
      sliceSizes.push_back(element.getSExtValue());
    }

    // RealDynamicSliceOp::start_indices is a 1-dimensional tensor.
    // DynamicSliceOp::start_indices is a vararg of 0-dimensional tensors.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<Value> startIndices;
    for (auto i = 0; i < static_cast<int64_t>(sliceSizes.size()); ++i) {
      auto startIndex1D = rewriter.create<SliceOp>(
          op.getLoc(), op.getStartIndices(), rewriter.getI64TensorAttr(i),
          rewriter.getI64TensorAttr(i + 1), rewriter.getI64TensorAttr(1));
      auto startIndex0DType = RankedTensorType::get(
          {},
          op.getStartIndices().getType().cast<ShapedType>().getElementType());
      auto startIndex0D = rewriter.create<ReshapeOp>(
          op.getLoc(), startIndex0DType, startIndex1D);
      startIndices.push_back(startIndex0D);
    }

    rewriter.replaceOpWithNewOp<DynamicSliceOp>(
        op, op.getOperand(), startIndices,
        rewriter.getI64TensorAttr(sliceSizes));
    return success();
  }
};

struct CanonicalizeRealDynamicSliceOpToSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> startIndices, limitIndices, strides;
    if (!succeeded(hlo::matchInts(op.getStartIndices(), startIndices)))
      return rewriter.notifyMatchFailure(op, "expected static start");
    if (!succeeded(hlo::matchInts(op.getLimitIndices(), limitIndices)))
      return rewriter.notifyMatchFailure(op, "expected static limit");
    if (!succeeded(hlo::matchInts(op.getStrides(), strides)))
      return rewriter.notifyMatchFailure(op, "expected static strides");
    rewriter.replaceOpWithNewOp<SliceOp>(
        op, op.getOperand(), rewriter.getI64TensorAttr(startIndices),
        rewriter.getI64TensorAttr(limitIndices),
        rewriter.getI64TensorAttr(strides));
    return success();
  }
};

struct StablehloCanonicalizeDynamismPass
    : public impl::StablehloCanonicalizeDynamismPassBase<
          StablehloCanonicalizeDynamismPass> {
  using StablehloCanonicalizeDynamismPassBase::
      StablehloCanonicalizeDynamismPassBase;

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 2;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::AnyOp;

    RewritePatternSet patterns(&getContext());
    patterns.add<CanonicalizeDynamicBroadcastInDimOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicConvOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicGatherOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicIotaOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicPadOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicReshapeOpPattern>(&getContext());
    patterns.add<CanonicalizeRealDynamicSliceOpToDynamicSliceOpPattern>(
        &getContext());
    patterns.add<CanonicalizeRealDynamicSliceOpToSliceOpPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace stablehlo
}  // namespace mlir
