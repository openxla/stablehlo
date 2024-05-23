/* Copyright 2024 The StableHLO Authors.
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

#include <cstdint>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOAGGRESSIVEFOLDERPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

// DenseElementsAttr can be constructed from ArrayRef<APInt> but not from
// ArrayRef<APSInt>. This helper bridges the gap.
DenseIntElementsAttr getTensorAttr(ShapedType type, ArrayRef<APSInt> values) {
  SmallVector<APInt> supportedValues(values);
  return DenseIntElementsAttr::get(type, supportedValues);
}

APSInt getAPSInt(Type type, uint64_t value) {
  unsigned numBits;
  bool isUnsigned;
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    numBits = integerType.getWidth();
    // Signless types are treated as signed, per StableHLO convention.
    isUnsigned = integerType.isUnsignedInteger();
  } else {
    llvm::report_fatal_error("expected integer type");
  }
  return APSInt({/*numBits=*/numBits, value},
                /*isUnsigned=*/isUnsigned);
}

LogicalResult validateResultTypeForEval(PatternRewriter& rewriter,
                                        Operation* op, ShapedType resultType) {
  if (!resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "unable to fold dynamically shaped result type to constant");
  return success();
}

// The patterns below implement partial evaluation of shape computations which
// is a critical part of implementing type refinement for ops like
// dynamic_broadcast_in_dim, dynamic_iota and dynamic_reshape whose shape
// depends on the value of their shape operands.

template <typename OpType, typename FuncType>
LogicalResult evalElementwise(PatternRewriter& rewriter, OpType op,
                              FuncType fn) {
  auto resultType = op.getType();
  if (failed(validateResultTypeForEval(rewriter, op, resultType)))
    return failure();

  if (!isa<IntegerType>(resultType.getElementType()))
    return rewriter.notifyMatchFailure(op,
                                       "expected integer result tensor type");

  SmallVector<APSInt> result;
  if constexpr (OpType::template hasTrait<OpTrait::OneOperand>()) {
    SmallVector<APSInt> operand;
    if (failed(hlo::matchInts(op.getOperand(), operand)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");
    for (const auto& operandEl : operand) {
      result.push_back(fn(operandEl));
    }
  } else if constexpr (OpType::template hasTrait<
                           OpTrait::NOperands<2>::Impl>()) {
    SmallVector<APSInt> lhs, rhs;
    if (failed(hlo::matchInts(op.getLhs(), lhs)) ||
        failed(hlo::matchInts(op.getRhs(), rhs)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");
    for (auto [lhsEl, rhsEl] : llvm::zip(lhs, rhs)) {
      result.push_back(fn(lhsEl, rhsEl));
    }
  } else if constexpr (OpType::template hasTrait<
                           OpTrait::NOperands<3>::Impl>()) {
    SmallVector<APSInt> x, y, z;
    if (failed(hlo::matchInts(op->getOperand(0), x)) ||
        failed(hlo::matchInts(op->getOperand(1), y)) ||
        failed(hlo::matchInts(op->getOperand(2), z)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");
    for (auto [xEl, yEl, zEl] : llvm::zip(x, y, z)) {
      result.push_back(fn(xEl, yEl, zEl));
    }
  } else {
    llvm::report_fatal_error("unsupported number of operands");
  }

  rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                          getTensorAttr(resultType, result));
  return success();
}

struct EvalAddOpPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs + rhs; });
  }
};

struct EvalAndOpPattern : public OpRewritePattern<AndOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AndOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "expected boolean element type");

    return evalElementwise(rewriter, op, [&](APSInt lhsInt, APSInt rhsInt) {
      return getAPSInt(resultType.getElementType(), lhsInt != 0 && rhsInt != 0);
    });
  }
};

struct EvalBroadcastInDimOpPattern : public OpRewritePattern<BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateResultTypeForEval(rewriter, op, resultType)))
      return failure();

    auto operandType = op.getOperand().getType();
    if (operandType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "expected 0-dimensional type");

    SmallVector<APSInt> operand;
    if (failed(hlo::matchInts(op.getOperand(), operand)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");
    auto scalar = operand[0];

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, getTensorAttr(op.getType(), scalar));
    return success();
  }
};

struct EvalClampOpPattern : public OpRewritePattern<ClampOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ClampOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt min, APSInt operand, APSInt max) {
                             if (operand < min) return min;
                             if (max < operand) return max;
                             return operand;
                           });
  }
};

struct EvalCompareOpPattern : public OpRewritePattern<CompareOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    return evalElementwise(rewriter, op, [&](APSInt lhs, APSInt rhs) {
      bool result;
      switch (op.getComparisonDirection()) {
        case ComparisonDirection::EQ:
          result = lhs == rhs;
          break;
        case ComparisonDirection::NE:
          result = lhs != rhs;
          break;
        case ComparisonDirection::GE:
          result = lhs >= rhs;
          break;
        case ComparisonDirection::GT:
          result = lhs > rhs;
          break;
        case ComparisonDirection::LE:
          result = lhs <= rhs;
          break;
        case ComparisonDirection::LT:
          result = lhs < rhs;
          break;
      }
      return getAPSInt(resultType.getElementType(), result);
    });
  }
};

struct EvalConcatenateOpPattern : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateResultTypeForEval(rewriter, op, resultType)))
      return failure();

    if (op.getDimension() != 0)
      return rewriter.notifyMatchFailure(op, "expected dimension = 0");

    SmallVector<APSInt> result;
    for (Value operand : op->getOperands()) {
      if (failed(hlo::matchInts(operand, result)))
        return rewriter.notifyMatchFailure(op, "expected constant operands");
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            getTensorAttr(resultType, result));
    return success();
  }
};

struct EvalConvertOpPattern : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateResultTypeForEval(rewriter, op, resultType)))
      return failure();

    if (!isa<IntegerType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "expected integer result tensor type with static shapes");
    auto resultBitWidth = resultType.getElementType().getIntOrFloatBitWidth();
    return evalElementwise(rewriter, op, [&](APSInt operand) {
      return operand.extOrTrunc(resultBitWidth);
    });
  }
};

struct EvalDivOpPattern : public OpRewritePattern<DivOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs / rhs; });
  }
};

struct EvalGetDimensionSizeOpPattern
    : public OpRewritePattern<GetDimensionSizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetDimensionSizeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateResultTypeForEval(rewriter, op, resultType)))
      return failure();

    auto operandType = op.getOperand().getType();
    if (operandType.isDynamicDim(op.getDimension()))
      return rewriter.notifyMatchFailure(op, "expected static dimension");

    auto result = operandType.getDimSize(op.getDimension());
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get<int32_t>(resultType, result));
    return success();
  }
};

struct EvalMaxOpPattern : public OpRewritePattern<MaxOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op, [&](APSInt lhs, APSInt rhs) {
      return lhs >= rhs ? lhs : rhs;
    });
  }
};

struct EvalMinOpPattern : public OpRewritePattern<MinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MinOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op, [&](APSInt lhs, APSInt rhs) {
      return lhs <= rhs ? lhs : rhs;
    });
  }
};

struct EvalMulOpPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs * rhs; });
  }
};

struct EvalOrOpPattern : public OpRewritePattern<OrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(OrOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "expected boolean element type");

    return evalElementwise(rewriter, op, [&](APSInt lhsInt, APSInt rhsInt) {
      return getAPSInt(resultType.getElementType(), lhsInt != 0 || rhsInt != 0);
    });
  }
};

struct EvalRemOpPattern : public OpRewritePattern<RemOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RemOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs % rhs; });
  }
};

struct EvalReshapeOpPattern : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateResultTypeForEval(rewriter, op, resultType)))
      return failure();

    DenseIntElementsAttr attr;
    if (!matchPattern(op.getOperand(), m_Constant(&attr)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");
    rewriter.replaceOpWithNewOp<ConstantOp>(op, attr.reshape(resultType));
    return success();
  }
};

struct EvalSelectOpPattern : public OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateResultTypeForEval(rewriter, op, resultType)))
      return failure();

    SmallVector<APSInt> pred, onTrue, onFalse;
    if (failed(hlo::matchInts(op.getPred(), pred)) ||
        failed(hlo::matchInts(op.getOnTrue(), onTrue)) ||
        failed(hlo::matchInts(op.getOnFalse(), onFalse)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");

    SmallVector<APSInt> result;
    for (auto [predEl, onTrueEl, onFalseEl] :
         llvm::zip(pred, onTrue, onFalse)) {
      result.push_back(predEl != 0 ? onTrueEl : onFalseEl);
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, getTensorAttr(op.getType(), result));
    return success();
  }
};

struct EvalSignOpPattern : public OpRewritePattern<SignOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SignOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (!isa<IntegerType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op,
                                         "expected integer result tensor type");
    return evalElementwise(rewriter, op, [&](APSInt operand) {
      int64_t result;
      if (operand.isNegative())
        result = -1;
      else if (operand.isZero())
        result = 0;
      else
        result = 1;
      return getAPSInt(resultType.getElementType(), result);
    });
  }
};

struct EvalSliceOpPattern : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateResultTypeForEval(rewriter, op, resultType)))
      return failure();

    if (resultType.getRank() < 1)
      return rewriter.notifyMatchFailure(
          op, "expected non-0 ranked tensor result type");

    auto operand = cast<TypedValue<RankedTensorType>>(op.getOperand());
    RankedTensorType operandType = operand.getType();
    if (!operandType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected operand with static ranked tensor type");

    // A ranked tensor type with unit dimension prefix of R-1 size is physically
    // compatible with 1-dimensional type.
    if (!llvm::all_of(resultType.getShape().drop_back(),
                      [](int64_t s) { return s == 1; }))
      return rewriter.notifyMatchFailure(
          op, "expected 1-dimensional compatible result type");

    SmallVector<APSInt> operandData;
    if (failed(hlo::matchInts(operand, operandData)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");

    const auto dimOffsets = computeSuffixProduct(operandType.getShape());
    auto startIndices = op.getStartIndices();
    auto limitIndices = op.getLimitIndices();
    auto strides = op.getStrides();

    int64_t start = 0;
    for (size_t i = 0; i < startIndices.size(); ++i)
      start += startIndices[i] * dimOffsets[i];

    auto slicedDim = operandType.getRank() - 1;
    int64_t limit = start + limitIndices[slicedDim] - startIndices[slicedDim];
    int64_t stride = strides[slicedDim];
    SmallVector<APSInt> result;
    for (auto i = start; i < limit; i += stride)
      result.push_back(operandData[i]);

    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            getTensorAttr(resultType, result));
    return success();
  }
};

struct EvalSubtractOpPattern : public OpRewritePattern<SubtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs - rhs; });
  }
};

struct EvalIotaOpPattern : public OpRewritePattern<IotaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IotaOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();

    if (!elementType.isInteger())
      return rewriter.notifyMatchFailure(op, "expected integer result type");

    auto outputSize = resultType.getNumElements();
    auto resultBitWidth = elementType.getIntOrFloatBitWidth();
    int64_t dimension = op.getIotaDimension();

    llvm::SmallVector<APInt> values;
    values.reserve(outputSize);

    if (outputSize == 0) {
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, DenseIntElementsAttr::get(resultType, values));
      return success();
    }

    int64_t sequences = 1;
    int64_t sequenceMax = resultType.getDimSize(dimension);
    int64_t elementRepetitions = 1;
    for (int64_t i = 0; i < resultType.getRank(); i++) {
      sequences *= i < dimension ? resultType.getDimSize(i) : 1;
      elementRepetitions *= i > dimension ? resultType.getDimSize(i) : 1;
    }

    for (int64_t i = 0; i < sequences; ++i) {
      for (int64_t value = 0; value < sequenceMax; ++value) {
        for (int64_t k = 0; k < elementRepetitions; ++k) {
          values.push_back(APInt(resultBitWidth, value));
        }
      }
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(resultType, values));
    return success();
  }
};

struct StablehloAggressiveFolderPass
    : public impl::StablehloAggressiveFolderPassBase<
          StablehloAggressiveFolderPass> {
  using StablehloAggressiveFolderPassBase::StablehloAggressiveFolderPassBase;

  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    populateStablehloAggressiveFolderPatterns(&patterns_, context);
    patterns = std::move(patterns_);

    return success();
  }

  void runOnOperation() override {
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patterns)))
      signalPassFailure();
  }

 private:
  FrozenRewritePatternSet patterns;
};

}  // namespace

void populateStablehloAggressiveFolderPatterns(RewritePatternSet* patterns,
                                               MLIRContext* context) {
  populateStablehloShapeFolderPatterns(patterns, context);
  patterns->add<EvalIotaOpPattern>(context);
}

void populateStablehloShapeFolderPatterns(RewritePatternSet* patterns,
                                          MLIRContext* context) {
  patterns->add<EvalAddOpPattern>(context);
  patterns->add<EvalAndOpPattern>(context);
  patterns->add<EvalBroadcastInDimOpPattern>(context);
  patterns->add<EvalClampOpPattern>(context);
  patterns->add<EvalCompareOpPattern>(context);
  patterns->add<EvalConcatenateOpPattern>(context);
  patterns->add<EvalConvertOpPattern>(context);
  patterns->add<EvalDivOpPattern>(context);
  patterns->add<EvalGetDimensionSizeOpPattern>(context);
  patterns->add<EvalMaxOpPattern>(context);
  patterns->add<EvalMinOpPattern>(context);
  patterns->add<EvalMulOpPattern>(context);
  patterns->add<EvalOrOpPattern>(context);
  patterns->add<EvalRemOpPattern>(context);
  patterns->add<EvalReshapeOpPattern>(context);
  patterns->add<EvalSelectOpPattern>(context);
  patterns->add<EvalSignOpPattern>(context);
  patterns->add<EvalSliceOpPattern>(context);
  patterns->add<EvalSubtractOpPattern>(context);
}

}  // namespace stablehlo
}  // namespace mlir
