// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements optional canonicalization patterns for StableHLO ops.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/optimization/Passes.h"

using llvm::SmallBitVector;

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOAGGRESSIVESIMPLIFICATIONPASS
#include "stablehlo/transforms/optimization/Passes.h.inc"

namespace {
// This is an upper limit on how many elements can be folded by an op folder.
// This limit doesn't apply to some special cases like adding a zero,
// multiplying by one, doing many operations with splats.
constexpr int64_t kFoldOpEltLimit = 65536;

static bool isIotaRange(ArrayRef<int64_t> dims) {
  return llvm::all_of(llvm::enumerate(dims), [](const auto &it) {
    return static_cast<int64_t>(it.index()) == it.value();
  });
}

/// Matches when either of the submatchers match.
template <typename MatcherA, typename MatcherB>
struct m_AnyOf {
  m_AnyOf(MatcherA a, MatcherB b) : matcherA(a), matcherB(b) {}

  bool match(Operation *op) { return matcherA.match(op) || matcherB.match(op); }

  MatcherA matcherA;
  MatcherB matcherB;
};

template <typename MatcherA, typename MatcherB>
m_AnyOf(MatcherA, MatcherB) -> m_AnyOf<MatcherA, MatcherB>;

/// Matches when either of the submatchers match.
template <typename MatcherA, typename MatcherB>
struct m_AnyAttrOf {
  m_AnyAttrOf(MatcherA a, MatcherB b) : matcherA(a), matcherB(b) {}

  bool match(Attribute attr) {
    return matcherA.match(attr) || matcherB.match(attr);
  }

  MatcherA matcherA;
  MatcherB matcherB;
};

template <typename MatcherA, typename MatcherB>
m_AnyAttrOf(MatcherA, MatcherB) -> m_AnyAttrOf<MatcherA, MatcherB>;

//////////////////////////////////
// CompareOp
/////////////////////////////////

static ComparisonDirection invertDirection(ComparisonDirection direction) {
  switch (direction) {
    case ComparisonDirection::EQ:
    case ComparisonDirection::NE:
      return direction;
    case ComparisonDirection::GE:
      return ComparisonDirection::LE;
    case ComparisonDirection::GT:
      return ComparisonDirection::LT;
    case ComparisonDirection::LE:
      return ComparisonDirection::GE;
    case ComparisonDirection::LT:
      return ComparisonDirection::GT;
  }

  llvm::report_fatal_error("Unhandled case");
}

struct CompareOpCanon final : OpRewritePattern<CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();

    // Bail out on non-integer comparison.
    // TODO: Support more comparison types.
    std::optional<ComparisonType> compType = op.getCompareType();
    if (!compType ||
        !llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            *compType)) {
      return failure();
    }

    ComparisonDirection direction = op.getComparisonDirection();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Pattern: compare(X, X, [EQ,GE,LE]) -> true
    // Pattern: compare(X, X, [NE,GT,LT]) -> false
    if (lhs == rhs) {
      switch (direction) {
        case ComparisonDirection::EQ:
        case ComparisonDirection::GE:
        case ComparisonDirection::LE: {
          rewriter.replaceOpWithNewOp<ConstantOp>(
              op, SplatElementsAttr::get(type, rewriter.getBoolAttr(true)));
          return success();
        }
        case ComparisonDirection::GT:
        case ComparisonDirection::LT:
        case ComparisonDirection::NE: {
          rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                                  rewriter.getZeroAttr(type));
          return success();
        }
      }
      llvm_unreachable("Unhandled case");
    }

    // Pattern: compare(cst, X, comparator) -> compare(X, cst, inv(comparator))
    TypedAttr lhsAttr, rhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (lhsAttr && !rhsAttr) {
      rewriter.modifyOpInPlace(op, [&op, direction, lhs, rhs] {
        op.setComparisonDirection(invertDirection(direction));
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    return failure();
  }
};

//////////////////////////////////
// ConcatenateOp
/////////////////////////////////

// Pattern: concatenate(X) -> X
class ConcatenateOpNoop : public OpRewritePattern<ConcatenateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 ||
        op.getInputs().front().getType() != op.getType())
      return rewriter.notifyMatchFailure(op, "not single operand noop-concat");

    rewriter.replaceOp(op, op.getInputs().front());
    return success();
  }
};

// Pattern: concatenate(X, Y, []) -> concatenate(X, Y)
class ConcatenateOpRemoveEmpty : public OpRewritePattern<ConcatenateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto axis = op.getDimension();
    llvm::SmallVector<Value> newOperands = llvm::to_vector(
        llvm::make_filter_range(op.getOperands(), [&](Value operand) {
          return cast<ShapedType>(operand.getType()).getDimSize(axis) != 0;
        }));

    // Only handle nonempty new operands, empty handled by
    // ZeroExtentToEmptyConstant pattern.
    if (!newOperands.empty() && newOperands.size() < op.getNumOperands()) {
      rewriter.modifyOpInPlace(op, [&] { op->setOperands(newOperands); });
      return success();
    }

    return failure();
  }
};

// Pattern: concatenate(concatenate(X, Y), Z) -> concatenate(X, Y, Z)
class ConcatenateOpFlatten : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto getFlattenedOperands = [&](const Value &val) -> ValueRange {
      auto definingOp = dyn_cast_or_null<ConcatenateOp>(val.getDefiningOp());
      // To avoid inflate the memory footprint, only flatten the
      // ConcatenateOp when it has only one use.
      if (definingOp && definingOp->hasOneUse() &&
          definingOp.getDimension() == op.getDimension())
        return definingOp.getInputs();
      return val;
    };

    bool needToFlatten = false;
    int operandCount = 0;
    llvm::for_each(op.getInputs(), [&](Value val) {
      auto result = getFlattenedOperands(val);
      if (result.size() != 1 || result[0] != val) needToFlatten = true;
      operandCount += result.size();
    });

    if (!needToFlatten)
      return rewriter.notifyMatchFailure(op, "no need to flatten");

    llvm::SmallVector<Value, 6> newOperands;
    newOperands.reserve(operandCount);

    for (auto operand : op.getInputs()) {
      auto flattenedOperands = getFlattenedOperands(operand);
      newOperands.append(flattenedOperands.begin(), flattenedOperands.end());
    }

    rewriter.modifyOpInPlace(op, [&] { op->setOperands(newOperands); });
    return success();
  }
};

//////////////////////////////////
// BroadcastInDimOp
/////////////////////////////////

// Used in DRR file.
DenseI64ArrayAttr getMergedBroadcastDimensions(OpBuilder &b,
                                               ArrayRef<int64_t> dims,
                                               ArrayRef<int64_t> dimsParent) {
  auto mergedDims = llvm::map_to_vector(
      dimsParent, [&dims](int64_t dim) { return dims[dim]; });
  return b.getDenseI64ArrayAttr(mergedDims);
}

//////////////////////////////////
// DynamicBroadcastInDimOp
/////////////////////////////////

/// Does the same as PatternRewriter::replaceOpWithNewOp, but with a twist.
///
/// Sometimes, we want to replace an op with a new op and simultaneously refine
/// the result type from a dynamically-shaped type to a statically-shaped type.
/// (Search for usages of this function for examples).
//
/// Oftentimes, this works just fine because HLO is designed to accommodate
/// this kind of type refinements. But sometimes, this doesn't work - when
/// the op is used outside of the HLO dialect (e.g. in func.return). In these
/// cases, we insert a stablehlo.convert to smooth things out.
template <typename OpTy, typename... Args>
static OpTy refineOpWithNewOp(PatternRewriter &rewriter, Operation *op,
                              Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);

  llvm::SmallVector<Value> replacementResults;
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  for (auto [opResult, newOpResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    Value replacementResult = newOpResult;
    if (llvm::any_of(opResult.getUsers(), [&](Operation *user) {
          return user->getDialect() != op->getDialect();
        }))
      replacementResult = rewriter.create<ConvertOp>(
          op->getLoc(), opResult.getType(), newOpResult);
    replacementResults.push_back(replacementResult);
  }

  rewriter.replaceOp(op, replacementResults);
  return newOp;
}

/// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
/// BroadcastInDimOp.
struct DynamicBroadcastInDimOpNotActuallyDynamic final
    : OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType operandType = op.getOperand().getType();
    if (!operandType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires operand static shape");

    RankedTensorType type = op.getType();
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, type, op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }

    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (llvm::SmallVector<int64_t> shape;
        succeeded(hlo::matchInts(op.getOutputDimensions(), shape))) {
      refineOpWithNewOp<BroadcastInDimOp>(
          rewriter, op, RankedTensorType::get(shape, type.getElementType()),
          op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

//////////////////////////////////
// DynamicGatherOp
/////////////////////////////////

DenseI64ArrayAttr convertToI64Array(OpBuilder &b, Attribute attr) {
  auto denseAttr = cast<ElementsAttr>(attr);
  SmallVector<int64_t> result;
  result.reserve(denseAttr.getNumElements());
  for (auto elem : denseAttr.getValues<APInt>())
    result.push_back(elem.getSExtValue());
  return b.getDenseI64ArrayAttr(result);
}

//////////////////////////////////
// DynamicIotaOp
/////////////////////////////////

struct DynamicIotaIsStatic : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter &rewriter) const override {
    // Result type has static shape, replace with iota.
    auto resultTy = cast<ShapedType>(iota.getType());
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(iota, "requires output static shape");
    rewriter.replaceOpWithNewOp<IotaOp>(iota, resultTy,
                                        iota.getIotaDimension());
    return success();
  }
};

// Dynamic Iota operations across multiple dimensions can be reduced to an iota
// and a ranked broadcast.
// Pattern: dynamic_iota(shape, dim) ->
//   dynamic_broadcast_in_dim(dynamic_iota(slice(shape), dim), shape)
struct DynamicIotaOpToBroadcast : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ShapedType>(iota.getType());
    if (resultTy.getRank() < 2)
      return rewriter.notifyMatchFailure(iota, "requires rank >= 2");

    auto iotaDimension = static_cast<int64_t>(iota.getIotaDimension());

    // Handle case where iota dimension is index, need to convert to/from i64
    // to interop with slice. These canonicalize away if input is i64.
    auto convertedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            cast<ShapedType>(iota.getOutputShape().getType()).getShape(),
            rewriter.getI64Type()),
        iota.getOutputShape());

    auto slicedShape = rewriter.create<SliceOp>(
        iota.getLoc(), convertedShape,
        rewriter.getDenseI64ArrayAttr(iotaDimension),
        rewriter.getDenseI64ArrayAttr(iotaDimension + 1),
        rewriter.getDenseI64ArrayAttr(1));

    auto convertedSlicedShape = rewriter.create<arith::IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            {1},
            cast<ShapedType>(iota.getOutputShape().getType()).getElementType()),
        slicedShape);

    auto iotaType = RankedTensorType::get({resultTy.getDimSize(iotaDimension)},
                                          resultTy.getElementType());

    auto newIota = rewriter.create<DynamicIotaOp>(
        iota.getLoc(), iotaType, convertedSlicedShape,
        rewriter.getI64IntegerAttr(0));

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        iota, resultTy, newIota, iota.getOutputShape(),
        rewriter.getDenseI64ArrayAttr(iotaDimension));
    return success();
  }
};

//////////////////////////////////
// DynamicReshapeOp
/////////////////////////////////

struct DynamicReshapeOpIsStatic final : OpRewritePattern<DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // This is a noop when the output type is already a static shape.
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic reshape not static");

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, type, op.getOperand());
    return success();
  }
};

// Pattern: dynamic_reshape(op(dynamic_reshape(X, shape)), shape)
//            -> op(dynamic_reshape(X, shape))
//            [if op has same operand and result shape]
class DynamicReshapeOpSameOperandAndResultShape
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = op.getOperand().getDefiningOp();
    if (!defOp ||
        !defOp->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic reshape parent not same operand and result shape");
    }
    DynamicReshapeOp reshape =
        defOp->getOperand(0).getDefiningOp<DynamicReshapeOp>();
    if (!reshape)
      return rewriter.notifyMatchFailure(
          op, "dynamic reshape not wrapping same operand and result shape");
    if (reshape.getOutputShape() == op.getOutputShape()) {
      rewriter.replaceOp(op, {defOp->getResult(0)});
      return success();
    }
    return failure();
  }
};

//////////////////////////////////
// DynamicSliceOp
/////////////////////////////////

// Canonicalizes DynamicSlice ops that can be replaced instead with Slice ops.
// This canonicalization is applied the case when the `begin` input values are
// compile time constants and thus can be made into a tensor.
//
// Pattern: dynamic_slice(X, begin, slice_sizes) -> slice(X, begin, slice_sizes)
struct DynamicSliceOpToSlice : public OpRewritePattern<DynamicSliceOp> {
  using OpRewritePattern<DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicSliceOp dynamicSlice,
                                PatternRewriter &rewriter) const override {
    Value input = dynamicSlice.getOperand();
    auto inputType = cast<ShapedType>(input.getType());
    if (!inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(dynamicSlice,
                                         "dynamic slice input not static");

    auto sliceSizes = dynamicSlice.getSliceSizes();
    SmallVector<int64_t, 4> tempStartIndices;
    for (const auto &indexAndSliceStart :
         llvm::enumerate(dynamicSlice.getStartIndices())) {
      APInt val;
      Value start = indexAndSliceStart.value();
      int64_t index = indexAndSliceStart.index();
      if (!matchPattern(start, m_ConstantInt(&val)))
        return rewriter.notifyMatchFailure(dynamicSlice,
                                           "dynamic slice input not constant");

      // Clamp the indices within bounds to faithfully mirror dynamic slice
      // semantics.
      int64_t clampedStart =
          std::clamp(val.getSExtValue(), static_cast<int64_t>(0),
                     inputType.getDimSize(index) - sliceSizes[index]);
      tempStartIndices.push_back(clampedStart);
    }

    // At this point we've determined that the start indices are all constants;
    // pack them into a single tensor.
    auto sliceStartIndices = rewriter.getDenseI64ArrayAttr(tempStartIndices);
    SmallVector<int64_t, 4> tempSliceLimits;
    for (const auto &[start, size] : llvm::zip(tempStartIndices, sliceSizes)) {
      tempSliceLimits.push_back(start + size);
    }
    auto sliceLimits = rewriter.getDenseI64ArrayAttr(tempSliceLimits);

    auto sliceStrides = rewriter.getDenseI64ArrayAttr(
        SmallVector<int64_t, 4>(inputType.getRank(), 1));

    rewriter.replaceOpWithNewOp<SliceOp>(dynamicSlice, input, sliceStartIndices,
                                         sliceLimits, sliceStrides);
    return success();
  }
};

//////////////////////////////////
// RealDynamicSliceOp
/////////////////////////////////

// Pattern: real_dynamic_slice(X, start, limit, strides)
//            -> dynamic_slice(X, start, limit, strides)
//            [if strides, start are constants, limit = start + constant]
struct RealDynamicSliceOpToDynamicSlice
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern<RealDynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
    // This rewrite only works for unit strides because DynamicSliceOp
    // doesn't support strides (i.e. it implicitly has unit strides).
    DenseIntElementsAttr stridesAttr;
    if (!matchPattern(op.getStrides(), m_Constant(&stridesAttr)))
      return rewriter.notifyMatchFailure(op, "requires constant strides");
    if (!llvm::all_of(stridesAttr.getValues<APInt>(),
                      [&](APInt stride) { return stride == 1; }))
      return rewriter.notifyMatchFailure(op, "requires unit strides");

    // Check that slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    // Only handle the AddOp case, if all constant we fold to SliceOp.
    if (!matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) &&
        !matchPattern(op.getLimitIndices(),
                      m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices)))
      return rewriter.notifyMatchFailure(
          op, "requires limit indices equal to start indices plus constant");

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
          op.getLoc(), op.getStartIndices(), rewriter.getDenseI64ArrayAttr(i),
          rewriter.getDenseI64ArrayAttr(i + 1),
          rewriter.getDenseI64ArrayAttr(1));
      auto startIndex0DType = RankedTensorType::get(
          {},
          cast<ShapedType>(op.getStartIndices().getType()).getElementType());
      auto startIndex0D = rewriter.create<ReshapeOp>(
          op.getLoc(), startIndex0DType, startIndex1D);
      startIndices.push_back(startIndex0D);
    }

    rewriter.replaceOpWithNewOp<DynamicSliceOp>(
        op, op.getOperand(), startIndices,
        rewriter.getDenseI64ArrayAttr(sliceSizes));
    return success();
  }
};

//////////////////////////////////
// ReduceOp
/////////////////////////////////

// Pattern: reduce[A](_, _, fn:return A) -> A...
struct ReduceOpNoopVariableReturn final : OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // If all returned values in the ReduceOp region exists outside the
    // region, replace the ReduceOp with those values.
    if (auto retOp = dyn_cast<ReturnOp>(op.getBody().front().getTerminator())) {
      Region *retRegion = retOp->getParentRegion();
      if (llvm::any_of(retOp.getResults(), [retRegion](Value result) {
            return result.getParentRegion() == retRegion;
          }))
        return failure();

      rewriter.replaceOp(op, retOp.getResults());
      return success();
    }

    return failure();
  }
};

// Pattern: reduce(empty_0, empty_1, ...) -> [broadcast_in_dim(empty_i)...]
struct ReduceOpEmptyCanon final : OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // We require all reduce shapes to be the same, up to the element types, so
    // we can just use the first operand and the first result as
    // representatives.
    auto elemTy = cast<RankedTensorType>(op.getInputs().getType().front());

    if (!llvm::is_contained(elemTy.getShape(), 0)) return failure();

    Location loc = op.getLoc();
    DenseI64ArrayAttr empty = rewriter.getDenseI64ArrayAttr({});
    if (elemTy.hasStaticShape()) {
      SmallVector<Value> broadcasts(op.getNumResults());
      for (auto [bcast, init, outTy] : llvm::zip_equal(
               broadcasts, op.getInitValues(), op.getResultTypes())) {
        bcast = rewriter.create<BroadcastInDimOp>(loc, outTy, init, empty);
      }
      rewriter.replaceOp(op, broadcasts);
      return success();
    }

    SmallVector<Value> shapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(), shapes)))
      return failure();

    SmallVector<Value> broadcasts(op.getNumResults());
    for (auto [bcast, init, shape, outTy] : llvm::zip_equal(
             broadcasts, op.getInitValues(), shapes, op.getResultTypes())) {
      bcast = rewriter.create<DynamicBroadcastInDimOp>(loc, outTy, init, shape,
                                                       empty);
    }
    rewriter.replaceOp(op, broadcasts);
    return success();
  }
};

// Pattern: reduce(in_1, in_2, _, _) -> reduce(in_1, _, _) [if unused(in_2)]
struct ReduceOpUnusedResultCanon final : OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpResult, 4> usedResults;
    llvm::copy_if(op.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    if (usedResults.size() == op.getNumResults())
      return rewriter.notifyMatchFailure(op, "all operation results have uses");

    const auto pairSize = 2;
    const auto numOperands = op.getNumOperands();
    const auto numOperandPairs = numOperands / pairSize;

    Block &reducerBlock = op.getBody().front();
    auto retOp = cast<ReturnOp>(reducerBlock.getTerminator());

    assert(numOperandPairs == op.getNumResults() &&
           numOperandPairs == retOp.getNumOperands());

    SmallVector<Value> workList;
    auto addToWorkList = [&workList,
                          reducerBody = retOp->getParentRegion()](Value v) {
      if (v.getParentRegion() == reducerBody) workList.push_back(v);
    };

    SmallPtrSet<Operation *, 16> usedOps;
    SmallBitVector usedArgs(numOperands);
    SmallBitVector usedReturnOperands(numOperandPairs);
    for (const auto &usedResult : usedResults) {
      auto resultNo = usedResult.getResultNumber();
      usedReturnOperands.set(resultNo);

      // Follow the def-use chain starting from return operand to identify
      // which argument pairs are used to compute it.
      addToWorkList(retOp.getOperand(resultNo));
      while (!workList.empty()) {
        auto definition = workList.pop_back_val();
        if (auto blockArg = dyn_cast<BlockArgument>(definition)) {
          // using one argument implies using the whole argument pair
          const auto pairNo = blockArg.getArgNumber() % numOperandPairs;
          usedArgs.set(pairNo);
          usedArgs.set(pairNo + numOperandPairs);
        } else if (auto *defOp = definition.getDefiningOp()) {
          usedOps.insert(defOp);
          for (const auto &operand : defOp->getOperands())
            addToWorkList(operand);
        }
      }
    }

    const auto newNumOperandPairs = usedResults.size();
    const auto newNumOperands = newNumOperandPairs * pairSize;
    if (newNumOperands != usedArgs.count())
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "non-conservative case: " << newNumOperandPairs
             << " return results should be matched with " << newNumOperands
             << " operands, but got " << usedArgs.count();
      });

    SmallVector<Value> newInputs;
    SmallVector<Value> newInitVals;
    SmallVector<Type> newElementTypes;
    for (auto i : llvm::seq(0u, numOperandPairs)) {
      if (usedReturnOperands[i])
        newElementTypes.push_back(
            getElementTypeOrSelf(retOp.getOperand(i).getType()));

      if (!usedArgs[i]) continue;

      newInputs.push_back(op.getOperand(i));
      newInitVals.push_back(op.getOperand(i + numOperandPairs));
    }

    auto newOp =
        rewriter.create<ReduceOp>(op.getLoc(), newInputs, newInitVals,
                                  op.getDimensionsAttr(), newElementTypes);
    Block *newReducerBlock = rewriter.createBlock(&newOp.getBody());

    IRMapping mapper;
    for (auto arg : reducerBlock.getArguments())
      if (usedArgs[arg.getArgNumber()])
        mapper.map(arg,
                   newReducerBlock->addArgument(arg.getType(), arg.getLoc()));

    rewriter.setInsertionPointToStart(newReducerBlock);
    for (Operation &op : reducerBlock.getOperations())
      if (usedOps.contains(&op)) rewriter.clone(op, mapper);

    SmallVector<Value> newReturnOperands;
    for (const auto &en : llvm::enumerate(retOp.getOperands()))
      if (usedReturnOperands[en.index()])
        newReturnOperands.push_back(mapper.lookup(en.value()));

    rewriter.create<ReturnOp>(retOp.getLoc(), newReturnOperands);

    // Build new results list (unused entries will be null).
    SmallVector<Value> newResults(op.getNumResults());
    for (const auto &[i, result] : llvm::enumerate(usedResults)) {
      newResults[result.getResultNumber()] = newOp.getResult(i);
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

/////////////////////////////////
// GetDimensionSizeOp
/////////////////////////////////

// TODO: This is duplicated with a pattern in shape refinement, consider
// consolidating.
// Pattern: get_dimension_size(X, i) -> X.shape[i]
struct GetDimensionSizeOpCanon final : OpRewritePattern<GetDimensionSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold get_dimension_size when the queried dim is statically known.
    RankedTensorType operandTy = op.getOperand().getType();

    int64_t dimSize = operandTy.getDimSize(op.getDimension());
    if (dimSize < 0) return failure();

    auto elemTy = cast<IntegerType>(op.getType().getElementType());
    IntegerAttr elemVal = rewriter.getIntegerAttr(elemTy, dimSize);
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), elemVal));
    return success();
  }
};

//////////////////////////////////
// GatherOp
/////////////////////////////////

/// Converts gather ops to slice ops in case we have a single set of constant
/// indices.
// Pattern: gather(X, cst_start_indices) -> slice(X, slice_start, slice_end)
struct GatherOpCanon final : OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index)))
      return failure();

    GatherDimensionNumbersAttr dnums = gather.getDimensionNumbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO: Remove when the verifier catches this case that is
    // invalid if all previous condition holds.
    if (index.getNumElements() !=
        static_cast<int64_t>(dnums.getStartIndexMap().size())) {
      return failure();
    }

    auto operandType = cast<RankedTensorType>(gather->getOperand(0).getType());
    if (!operandType.hasStaticShape()) return failure();

    auto sliceEnd = llvm::to_vector(gather.getSliceSizes());
    SmallVector<int64_t> sliceStart(sliceEnd.size(), 0);
    for (auto [mapIndex, value] :
         llvm::zip_equal(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset =
          std::clamp(value.getSExtValue(), static_cast<int64_t>(0),
                     operandType.getDimSize(mapIndex) - sliceEnd[mapIndex]);
      sliceStart[mapIndex] += offset;
      sliceEnd[mapIndex] += offset;
    }

    SmallVector<int64_t> sliceStride(sliceEnd.size(), 1);
    SmallVector<int64_t> sliceShape(sliceEnd.size());
    for (auto [shapeElem, startElem, endElem] :
         llvm::zip_equal(sliceShape, sliceStart, sliceEnd)) {
      shapeElem = endElem - startElem;
    }

    Type elementType = gather.getType().getElementType();
    auto sliceType = RankedTensorType::get(sliceShape, elementType);
    Value result = rewriter.create<SliceOp>(
        gather.getLoc(), sliceType, gather.getOperand(),
        rewriter.getDenseI64ArrayAttr(sliceStart),
        rewriter.getDenseI64ArrayAttr(sliceEnd),
        rewriter.getDenseI64ArrayAttr(sliceStride));

    ArrayRef<int64_t> collapsedSliceDims = dnums.getCollapsedSliceDims();
    if (!collapsedSliceDims.empty()) {
      llvm::SmallVector<int64_t> reshapeShape;
      for (auto [idx, dim] : llvm::enumerate(sliceShape)) {
        if (!llvm::is_contained(collapsedSliceDims, idx))
          reshapeShape.push_back(dim);
      }
      auto reshapeType = RankedTensorType::get(reshapeShape, elementType);
      result = rewriter.create<ReshapeOp>(gather.getLoc(), reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

//////////////////////////////////
// IotaOp
/////////////////////////////////

// Iota operations across multiple dimensions can be reduced to an iota and a
// ranked broadcast.
// Pattern: iota(dim) : multi_rank
//            -> broadcast_in_dim(iota(dim) : array, multi_rank)
struct IotaOpBroadcast : public OpRewritePattern<IotaOp> {
  using OpRewritePattern<IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp iota,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ShapedType>(iota.getType());
    if (resultTy.getRank() < 2)
      return rewriter.notifyMatchFailure(iota, "itoa not broadcastable");

    auto iotaDim = iota.getIotaDimension();
    auto iotaDimSize = resultTy.getDimSize(iotaDim);
    auto iota1D = rewriter.create<IotaOp>(
        iota.getLoc(),
        RankedTensorType::get({iotaDimSize}, resultTy.getElementType()),
        rewriter.getI64IntegerAttr(0));

    auto broadcastAttr =
        rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(iotaDim)});
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iota, resultTy, iota1D,
                                                  broadcastAttr);
    return success();
  }
};

//////////////////////////////////
// PadOp
/////////////////////////////////

// If the input tensor has a dimension of length-0, the input tensor is
// irrelevant. Instead we can broadcast the pad value to the output size rather
// than pad the input tensor.

// If the input tensor has a dimension of length-0, the input tensor is
// irrelevant. Instead we can broadcast the pad value to the output size rather
// than pad the input tensor.

// Pattern: pad(empty_tensor, _) -> broadcast_in_dim(empty_tensor, _)
struct PadOpBroadcastEmptyTensor : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();
    auto padVal = op.getPaddingValue();

    auto resultTy = cast<RankedTensorType>(op.getType());

    if (cast<ShapedType>(operand.getType()).getNumElements() != 0)
      return rewriter.notifyMatchFailure(op, "operand is not empty tensor");

    if (resultTy.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
          op, resultTy, padVal, rewriter.getDenseI64ArrayAttr({}));
      return success();
    }

    llvm::SmallVector<Value> reifiedShapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(),
                                        reifiedShapes)))
      return rewriter.notifyMatchFailure(op, "failed to reify return type");

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        op, op.getType(), padVal, reifiedShapes.front(),
        rewriter.getDenseI64ArrayAttr({}));
    return success();
  }
};

//////////////////////////////////
// SelectOp
/////////////////////////////////

struct SelectOpCanon final : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();

    Value trueVal = op.getOnTrue();
    Value falseVal = op.getOnFalse();

    // Eliminate select with two identical outcomes.
    if (trueVal == falseVal) {
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    // Simplify when the condition is a constant.
    Value pred = op.getPred();
    ElementsAttr cond;
    if (!matchPattern(pred, m_Constant(&cond))) return failure();

    // Handle splat predicate and select either `trueVal` or `falseVal`.
    if (cond.isSplat()) {
      rewriter.replaceOp(op, cond.getSplatValue<bool>() ? trueVal : falseVal);
      return success();
    }

    // Handle elementwise selection when both outcomes are also constants. This
    // will create a new, likely non-splat constant.
    if (cond.getNumElements() > kFoldOpEltLimit) return failure();

    ElementsAttr trueAttr;
    if (!matchPattern(trueVal, m_Constant(&trueAttr))) return failure();

    ElementsAttr falseAttr;
    if (!matchPattern(falseVal, m_Constant(&falseAttr))) return failure();

    SmallVector<Attribute> newValues;
    newValues.reserve(cond.getNumElements());
    for (auto [condElem, trueElem, falseElem] : llvm::zip_equal(
             cond.getValues<bool>(), trueAttr.getValues<Attribute>(),
             falseAttr.getValues<Attribute>())) {
      newValues.push_back(condElem ? trueElem : falseElem);
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseElementsAttr::get(type, newValues));
    return success();
  }
};

struct CompareSelectIntoMinMax final : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value pred = op.getPred();
    Value trueVal = op.getOnTrue();
    Value falseVal = op.getOnFalse();

    auto cmpOp = pred.getDefiningOp<CompareOp>();
    if (!cmpOp) return failure();

    ComparisonDirection direction = cmpOp.getComparisonDirection();
    Value cmpLhs = cmpOp.getLhs();
    Value cmpRhs = cmpOp.getRhs();

    // Turn into canonical form:
    // b <= a ? a : b  ---> a >= b ? a : b
    // b <  a ? a : b  ---> a >  b ? a : b
    // b >= a ? a : b  ---> a <= b ? a : b
    // b >  a ? a : b  ---> a <  b ? a : b
    if (cmpLhs == falseVal && cmpRhs == trueVal) {
      direction = invertDirection(direction);
    } else if (!(cmpLhs == trueVal && cmpRhs == falseVal)) {
      return failure();
    }

    switch (direction) {
      case ComparisonDirection::GE:
      case ComparisonDirection::GT: {
        rewriter.replaceOpWithNewOp<MaxOp>(op, trueVal, falseVal);
        return success();
      }
      case ComparisonDirection::LE:
      case ComparisonDirection::LT: {
        rewriter.replaceOpWithNewOp<MinOp>(op, trueVal, falseVal);
        return success();
      }
      default: {
        return failure();
      }
    }
  }
};

//////////////////////////////////
// SliceOp
/////////////////////////////////

// In cases where a concat is fed into a slice, it is possible the concat
// can be simplified or bypassed. This checks which inputs to the concat are
// used by the slice, either reducing the number of concatenated values or
// entirely removes the concat.
// Pattern: slice(concat(X,Y,Z,...),...) -> concat(slice(X),slice(Y),slice(Z))
struct SliceOpConcatSimplify : public OpRewritePattern<SliceOp> {
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp slice,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ShapedType>(slice.getType());
    if (!resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(slice, "result shape not static");

    auto concat = slice.getOperand().getDefiningOp<ConcatenateOp>();
    if (!concat)
      return rewriter.notifyMatchFailure(slice, "slice input not concat");

    auto concatType = cast<ShapedType>(concat.getType());
    auto dimension = concat.getDimension();

    auto start = slice.getStartIndices();
    auto limit = slice.getLimitIndices();

    int64_t sliceStart = start[dimension];
    int64_t sliceLimit = limit[dimension];

    // We need to determine what inputs from the concat affect the slice, and
    // how the bounds of the slice need to be updated for the minimally required
    // inputs.
    int64_t runningSize = 0;
    int64_t frontOffset = concatType.getShape()[dimension];

    auto subsetStart = concat.operand_end();
    auto subsetEnd = concat.operand_end();
    for (auto it = concat.operand_begin(); it < concat.operand_end(); ++it) {
      auto input = *it;
      ShapedType inputTy = cast<ShapedType>(input.getType());
      if (inputTy.isDynamicDim(dimension))
        return rewriter.notifyMatchFailure(
            slice, "concat input has dynamic dimension");

      auto dimSize = inputTy.getShape()[dimension];

      // If this position is in the slice its the start of the subset and we
      // need to update the start and limit values.
      if (runningSize + dimSize > sliceStart &&
          subsetStart == concat.operand_end()) {
        subsetStart = it;
        frontOffset = runningSize;
      }

      // Determine the last required offset.
      if (runningSize < sliceLimit) {
        subsetEnd = it + 1;
      }

      runningSize += dimSize;
    }

    auto subsetSize = subsetEnd - subsetStart;
    // We need all inputs so no optimization.
    if (subsetSize == concat.getNumOperands())
      return rewriter.notifyMatchFailure(slice,
                                         "slice needs all concat inputs");

    // If there's nothing to slice that means the output is an empty tensor and
    // there is dead code. We do nothing here and rely on other passes to clean
    // this up.
    if (subsetSize == 0)
      return rewriter.notifyMatchFailure(slice, "slice is empty");

    if (subsetSize > 1 && !concat.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(slice,
                                         "slice is not the only concat user");

    auto concatRange = OperandRange(subsetStart, subsetEnd);
    auto newConcat = rewriter.create<ConcatenateOp>(
        concat.getLoc(), concatRange, concat.getDimension());

    SmallVector<int64_t> newStart(start);
    SmallVector<int64_t> newLimit(limit);
    newStart[dimension] -= frontOffset;
    newLimit[dimension] -= frontOffset;

    rewriter.replaceOpWithNewOp<SliceOp>(
        slice, newConcat, rewriter.getDenseI64ArrayAttr(newStart),
        rewriter.getDenseI64ArrayAttr(newLimit), slice.getStrides());
    return success();
  }
};

//////////////////////////////////
// SortOp
/////////////////////////////////

/// Drops the operands if the results are not used and they are not used in
/// op.comparator().

// Pattern: sort(X,Y) -> sort(X) [if Y unused and unused in comparator]
struct SortOpDropUnusedArgs : public OpRewritePattern<SortOp> {
  using OpRewritePattern<SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    DenseSet<unsigned> erasedArgs;
    unsigned numOperands = op.getNumOperands();
    for (unsigned i = 0; i < numOperands; ++i) {
      if (!op.getResult(i).use_empty()) continue;
      Block &block = op.getComparator().front();
      if (!block.getArgument(i * 2).use_empty()) continue;
      if (!block.getArgument(i * 2 + 1).use_empty()) continue;
      erasedArgs.insert(i);
    }
    if (erasedArgs.empty()) return failure();

    SmallVector<Value> newOperands;
    BitVector erasedBlockArgs(op.getNumOperands() * 2);
    for (const auto &en : llvm::enumerate(op.getInputs())) {
      if (erasedArgs.contains(en.index())) {
        erasedBlockArgs.set(en.index() * 2);
        erasedBlockArgs.set(en.index() * 2 + 1);
      } else {
        newOperands.push_back(en.value());
      }
    }

    auto newOp = rewriter.create<SortOp>(op.getLoc(), newOperands,
                                         op.getDimension(), op.getIsStable());
    Region &region = newOp.getComparator();
    rewriter.inlineRegionBefore(op.getComparator(), region, region.end());
    region.front().eraseArguments(erasedBlockArgs);

    SmallVector<Value> results;
    for (unsigned i = 0, j = 0; i < numOperands; ++i) {
      if (erasedArgs.contains(i)) {
        results.push_back({});
      } else {
        results.push_back(newOp.getResult(j++));
      }
    }
    rewriter.replaceOp(op, results);

    return success();
  }
};

/// Set the sorting dimension to the last dimension if it's not set and the rank
/// is known.
// Pattern: sort(X) -> sort(X, dim = N) [when dim can be inferred]
struct SortOpSetDimension : public OpRewritePattern<SortOp> {
  using OpRewritePattern<SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResults().empty() ||
        static_cast<int64_t>(op.getDimension()) != -1)
      return rewriter.notifyMatchFailure(op,
                                         "dimension already set or no results");

    auto type = cast<ShapedType>(op.getResultTypes()[0]);
    IntegerAttr dim = rewriter.getI64IntegerAttr(type.getRank() - 1);
    auto newOp =
        rewriter.create<SortOp>(op.getLoc(), op.getResultTypes(),
                                op.getInputs(), dim, op.getIsStableAttr());
    newOp.getComparator().takeBody(op.getComparator());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

//////////////////////////////////
// TransposeOp
/////////////////////////////////

// Pattern: transpose(X, [no_mem_layout_change...]) -> reshape(X)
struct TransposeIsReshape final : OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto permutation = op.getPermutation();

    RankedTensorType inputTy = input.getType();
    if (!inputTy.hasStaticShape() || !op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "requires input and output to be of a statically-shaped ranked "
          "tensor type");

    // Check that the permutation is a valid memory layout change.
    // All non-zero/one dimensions must be in increasing order.
    SmallVector<int64_t> nonZeroPerms;
    nonZeroPerms.reserve(permutation.size());
    for (auto idx : permutation)
      if (inputTy.getDimSize(idx) != 1) nonZeroPerms.push_back(idx);

    for (size_t i = 1; i < nonZeroPerms.size(); ++i)
      if (nonZeroPerms[i - 1] > nonZeroPerms[i])
        return rewriter.notifyMatchFailure(op, "memory layout change");

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), input);
    return success();
  }
};

//////////////////////////////////
// TupleOp
/////////////////////////////////

// Pattern: tuple(get_tuple_element(X, 0), get_tuple_element(X, 1), ...) -> X
struct TupleIsRepacking : public OpRewritePattern<TupleOp> {
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getVal().empty())
      return rewriter.notifyMatchFailure(op, "empty tuple");

    // Get parent tuple
    Value firstElement = op.getVal().front();
    auto firstElementOp = firstElement.getDefiningOp<GetTupleElementOp>();
    if (!firstElementOp)
      return rewriter.notifyMatchFailure(op, "parent not get_tuple_element");

    Value tuplePredecessor = firstElementOp.getOperand();
    if (tuplePredecessor.getType() != op.getType())
      return rewriter.notifyMatchFailure(
          op, "tuple predecessor type does not match");

    // Check that this is a repacking of the parent tuple.
    for (const auto &elementAndIdx : llvm::enumerate(op.getVal())) {
      auto elementOp = elementAndIdx.value().getDefiningOp<GetTupleElementOp>();
      if (!elementOp ||
          elementOp.getIndexAttr().getInt() !=
              static_cast<int64_t>(elementAndIdx.index()) ||
          elementOp.getOperand() != tuplePredecessor)
        return rewriter.notifyMatchFailure(
            op, "not a repacking of the parent tuple");
    }

    rewriter.replaceOp(op, tuplePredecessor);
    return success();
  }
};

/////////////////////////////////
// WhileOp
/////////////////////////////////

// Turn loop invariant values into implicit capture.
// Check if there is at least one value is forwarded from one iteration to
// the next, or one of the yielded value is an implicit capture already.
// Otherwise there is nothing to do here.

// Pattern: while -> while (loop invariants as implicit captures)
struct WhileOpImplicitCapture : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Block *cond = whileOp.SingleBlock::getBody(0);
    Block *body = whileOp.SingleBlock::getBody(1);
    auto bodyReturnOp = cast<ReturnOp>(body->getTerminator());
    if (!llvm::any_of(llvm::zip(whileOp->getOperands(), body->getArguments(),
                                bodyReturnOp->getOperands()),
                      [&](auto zip) {
                        return (std::get<0>(zip) == std::get<2>(zip) ||
                                std::get<1>(zip) == std::get<2>(zip));
                      }))
      return rewriter.notifyMatchFailure(whileOp, "no loop invariant found");

    SmallVector<Value> newOperands, resultsToReplace;
    SmallVector<unsigned> invariantArgIdxs;
    BitVector invariantArgIdxBitVector(cond->getNumArguments());
    for (const auto &enumeratedOperands : llvm::enumerate(llvm::zip(
             whileOp.getOperands(), cond->getArguments(), body->getArguments(),
             bodyReturnOp->getOperands(), whileOp->getResults()))) {
      const auto &operands = enumeratedOperands.value();
      Value whileOperand = std::get<0>(operands);
      BlockArgument condBlockArg = std::get<1>(operands);
      BlockArgument bodyBlockArg = std::get<2>(operands);
      Value bodyReturnOperand = std::get<3>(operands);
      Value whileResult = std::get<4>(operands);

      bool forwarded = (whileOperand == bodyReturnOperand ||
                        bodyBlockArg == bodyReturnOperand);
      if (forwarded) {
        invariantArgIdxs.push_back(enumeratedOperands.index());
        invariantArgIdxBitVector.set(enumeratedOperands.index());
        condBlockArg.replaceAllUsesWith(whileOperand);
        bodyBlockArg.replaceAllUsesWith(whileOperand);
        whileResult.replaceAllUsesWith(whileOperand);
        continue;
      }
      newOperands.push_back(whileOperand);
      resultsToReplace.push_back(whileResult);
    }
    cond->eraseArguments(invariantArgIdxBitVector);
    body->eraseArguments(invariantArgIdxBitVector);
    for (int idx : llvm::reverse(invariantArgIdxs))
      bodyReturnOp->eraseOperand(idx);

    WhileOp newWhileOp = rewriter.create<WhileOp>(
        whileOp.getLoc(), bodyReturnOp->getOperandTypes(), newOperands);
    newWhileOp.getBodyRegion(0).takeBody(whileOp.getBodyRegion(0));
    newWhileOp.getBodyRegion(1).takeBody(whileOp.getBodyRegion(1));
    for (auto results : llvm::zip(resultsToReplace, newWhileOp->getResults()))
      std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
    rewriter.eraseOp(whileOp);
    return success();
  }
};

//////////////////////////////////
// Generic and Elementwise Ops
/////////////////////////////////

/// Check if a `t` is a tensor with zero extents.
static std::optional<RankedTensorType> getMaybeZeroExtentType(Type t) {
  auto type = dyn_cast<RankedTensorType>(t);
  if (type && type.hasStaticShape() && type.getNumElements() == 0) return type;
  return std::nullopt;
}

// Replace instances of zero extent tensors with empty tensors
// Pattern: op(X : zero_extent_tensor) -> constant([])
struct ZeroExtentToEmptyConstant final : RewritePattern {
  ZeroExtentToEmptyConstant(MLIRContext *context, PatternBenefit benefit)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (!isa_and_present<StablehloDialect>(op->getDialect()))
      return rewriter.notifyMatchFailure(op, "not stablehlo");
    if (isa<ConstantOp>(op))
      return rewriter.notifyMatchFailure(op, "op is empty constant");

    // If the result is a zero-extent tensor, replace the whole op with an empty
    // constant.
    bool didUpdate = false;
    for (auto result : op->getResults()) {
      auto resultType = getMaybeZeroExtentType(result.getType());
      if (!resultType || result.use_empty()) continue;
      rewriter.replaceAllUsesWith(
          result, rewriter.create<ConstantOp>(
                      loc, result.getType(),
                      DenseElementsAttr::get(resultType.value(),
                                             ArrayRef<Attribute>())));
      didUpdate = true;
    }

    // If one of the operands is a zero-extent tensor, replace the operand with
    // an empty tensor.
    for (OpOperand &operand : op->getOpOperands()) {
      auto operandType = getMaybeZeroExtentType(operand.get().getType());
      if (!operandType || operand.get().getDefiningOp<ConstantOp>()) continue;
      Operation *owner = operand.getOwner();
      int operandNum = operand.getOperandNumber();
      auto emptyConstantOp = rewriter.create<ConstantOp>(
          loc, operandType.value(),
          DenseElementsAttr::get(operandType.value(), ArrayRef<Attribute>()));
      rewriter.modifyOpInPlace(
          owner, [&]() { owner->setOperand(operandNum, emptyConstantOp); });
      didUpdate = true;
    }
    return success(didUpdate);
  }
};

struct ReorderElementwiseAndShapeOp final
    : OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern::OpTraitRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getOperands().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected to be unary");

    auto definingOp = op->getOperand(0).getDefiningOp();
    if (!definingOp)
      return rewriter.notifyMatchFailure(
          op, "expected to have an op before elementise op");

    if (!isa<ReshapeOp, TransposeOp, BroadcastOp>(definingOp))
      return rewriter.notifyMatchFailure(
          op, "defining operation of unexpected type");

    // Reshape and broadcast are not allowed to have dynamic shape.
    Value result = op->getResult(0);
    if (isa<ReshapeOp, BroadcastOp>(definingOp) &&
        !cast<ShapedType>(result.getType()).hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "cannot reorder around reshape/broadcast with dynamic shape");

    // Only reorder if the defining op has no other uses.
    if (!llvm::hasSingleElement(definingOp->getResult(0).getUses()))
      return rewriter.notifyMatchFailure(op, "operation has more than one use");

    Value input = definingOp->getOperand(0);
    auto intermediateType = cast<ShapedType>(input.getType())
                                .clone(getElementTypeOrSelf(result.getType()));

    // Reorder the operation and rewire the inputs/outputs.
    op->moveBefore(definingOp);
    definingOp->getResult(0).setType(result.getType());
    rewriter.replaceAllUsesWith(result, definingOp->getResult(0));
    result.setType(intermediateType);
    op->setOperands(input);
    definingOp->setOperands(result);
    return success();
  }
};

// Fuses batch normalization operation with convolution weight:
// X = conv(input, weight)
// Y = batch_norm_inference(X, ...)
// into ->
// X = conv(input, weight(new))
// Y = add(X, broadcast_in_dim(Bias(new)))
//
struct FuseConvolutionBatchNormalization final
    : OpRewritePattern<BatchNormInferenceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormInferenceOp op,
                                PatternRewriter &rewriter) const override {
    auto bnOperandType = op.getOperand().getType();
    auto bnOperandShape = bnOperandType.getShape();
    auto bnResultType = op.getResult().getType();
    uint64_t bnFeatureIndex = op.getFeatureIndex();

    auto convOp = op.getOperand().getDefiningOp<ConvolutionOp>();
    if (!convOp) return failure();

    auto convWeight = convOp.getRhs();
    auto convWeightType = convWeight.getType();
    auto convWeightShape = convWeightType.getShape();

    auto dimNumbers = convOp.getDimensionNumbers();
    if (dimNumbers.getInputBatchDimension() != 0 ||
        dimNumbers.getInputFeatureDimension() != 1 ||
        dimNumbers.getOutputBatchDimension() != 0 ||
        dimNumbers.getOutputFeatureDimension() != 1 ||
        dimNumbers.getKernelOutputFeatureDimension() != 0 ||
        dimNumbers.getKernelInputFeatureDimension() != 1)
      return rewriter.notifyMatchFailure(convOp,
                                         "Only [b, f, ...]x[o, i, ...]->[b, f, "
                                         "...] configuration is supported");

    if (convOp.getFeatureGroupCount() > 1 || convOp.getBatchGroupCount() > 1)
      return rewriter.notifyMatchFailure(
          convOp, "feature or batch grouping is not supported");

    if (bnOperandShape[bnFeatureIndex] != convWeightShape.front())
      return failure();

    DenseFPElementsAttr convWeightElems;
    DenseFPElementsAttr scaleElems;
    DenseFPElementsAttr offsetElems;
    DenseFPElementsAttr meanElems;
    DenseFPElementsAttr varianceElems;

    auto epsilon = op.getEpsilon();

    if (!matchPattern(convWeight, m_Constant(&convWeightElems)))
      return rewriter.notifyMatchFailure(
          op, "expected constant convolution weight");

    if (!matchPattern(op.getScale(), m_Constant(&scaleElems)) ||
        !matchPattern(op.getOffset(), m_Constant(&offsetElems)) ||
        !matchPattern(op.getMean(), m_Constant(&meanElems)) ||
        !matchPattern(op.getVariance(), m_Constant(&varianceElems)))
      return failure();

    const auto &convWeightSemantics =
        cast<FloatType>(convWeightType.getElementType()).getFloatSemantics();

    // W(new) = W(old) * gamma * rsqrt(variance + epsilon)
    // B(new) = (B(old) - mean) * rsqrt(variance + epsilon) * gamma + betta
    // where: gamma - scaling factor
    //        betta - shifting factor
    //        rsqrt - reciprocal square root function
    //        W - weight
    //        B - bias
    //
    const SmallVector<double> multipliers = llvm::map_to_vector(
        llvm::zip_equal(varianceElems, scaleElems),
        [&epsilon](const std::tuple<APFloat, APFloat> &pack) -> double {
          const auto &[variance, scale] = pack;
          auto varEps = (variance + epsilon).convertToDouble();
          auto rsqrt = 1.0 / std::sqrt(varEps);
          return rsqrt * scale.convertToDouble();
        });

    SmallVector<APFloat> newWeight;
    newWeight.reserve(convWeightType.getNumElements());

    const size_t outFeatureTileSize =
        computeProduct(convWeightShape.drop_front());
    auto it = convWeightElems.begin();
    for (const auto &multiplier : multipliers) {
      for (size_t i = 0; i < outFeatureTileSize; ++i) {
        double v = (*it).convertToDouble() * multiplier;
        APFloat result(v);
        bool losesInfo;
        if (APFloat::opStatus::opInvalidOp ==
            result.convert(convWeightSemantics, APFloat::rmNearestTiesToEven,
                           &losesInfo))
          return failure();
        newWeight.push_back(result);
        ++it;
      }
    }

    SmallVector<APFloat> biasValues;
    biasValues.reserve(multipliers.size());

    for (const auto &[off, multiplier, mean] :
         llvm::zip_equal(offsetElems, multipliers, meanElems)) {
      // stablehlo convolution operation doesn't have a builtin bias
      double convBias = 0;
      double v = (convBias - mean.convertToDouble()) * multiplier +
                 off.convertToDouble();
      APFloat result(v);

      bool losesInfo;
      if (APFloat::opStatus::opInvalidOp ==
          result.convert(convWeightSemantics, APFloat::rmNearestTiesToEven,
                         &losesInfo))
        return failure();

      biasValues.push_back(result);
    }

    rewriter.setInsertionPoint(op);
    auto newConvWeight = rewriter.create<ConstantOp>(
        convWeight.getLoc(), convWeightType,
        DenseFPElementsAttr::get(convWeightType, newWeight));

    // Keep old convolution as it might have other users
    auto newConvOp = rewriter.create<ConvolutionOp>(
        convOp.getLoc(), convOp->getResultTypes(),
        ValueRange{convOp.getLhs(), newConvWeight}, convOp->getAttrs());

    SmallVector<int64_t> biasShape{static_cast<int64_t>(biasValues.size())};
    auto biasType =
        convWeightType.cloneWith(biasShape, convWeightType.getElementType());
    auto bias = rewriter.create<ConstantOp>(
        op.getLoc(), biasType, DenseFPElementsAttr::get(biasType, biasValues));

    auto indices =
        rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(bnFeatureIndex)});
    auto bcast = rewriter.create<BroadcastInDimOp>(op.getLoc(), bnResultType,
                                                   bias, indices);
    auto add = rewriter.create<AddOp>(op.getLoc(), newConvOp, bcast);

    rewriter.replaceOp(op, add);
    return success();
  }
};

struct StablehloAggressiveSimplificationPass final
    : impl::StablehloAggressiveSimplificationPassBase<
          StablehloAggressiveSimplificationPass> {
  StablehloAggressiveSimplificationPass() = default;
  StablehloAggressiveSimplificationPass(GreedyRewriteConfig config)
      : config(config) {}
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet patterns_(context);
    populateStablehloCanonicalizationPatterns(context, &patterns_);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    if (failed(applyPatternsGreedily(getOperation(), patterns, config)))
      signalPassFailure();
  }

 private:
  GreedyRewriteConfig config;
  FrozenRewritePatternSet patterns;
};

#include "stablehlo/transforms/optimization/StablehloAggressiveSimplificationPatterns.h.inc"
}  // namespace

void populateStablehloCanonicalizationPatterns(MLIRContext *context,
                                               RewritePatternSet *patterns,
                                               PatternBenefit benefit) {
  populateWithGenerated(*patterns);
  patterns->add<ReorderElementwiseAndShapeOp>(context);
  patterns->add<
      CompareOpCanon, CompareSelectIntoMinMax, ConcatenateOpFlatten,
      ConcatenateOpNoop, ConcatenateOpRemoveEmpty, DynamicIotaOpToBroadcast,
      DynamicReshapeOpSameOperandAndResultShape, DynamicSliceOpToSlice,
      GatherOpCanon, IotaOpBroadcast, PadOpBroadcastEmptyTensor,
      RealDynamicSliceOpToDynamicSlice, ReduceOpEmptyCanon,
      ReduceOpNoopVariableReturn, ReduceOpUnusedResultCanon, SelectOpCanon,
      SliceOpConcatSimplify, SortOpDropUnusedArgs, SortOpSetDimension,
      TransposeIsReshape, TupleIsRepacking, WhileOpImplicitCapture>(context,
                                                                    benefit);

  // Generic patterns
  patterns->add<ReorderElementwiseAndShapeOp, ZeroExtentToEmptyConstant>(
      context, benefit);

  // TODO: Dynamism Refinements, consider merging with canonicalize dynamism
  patterns
      ->add<GetDimensionSizeOpCanon, DynamicBroadcastInDimOpNotActuallyDynamic,
            DynamicReshapeOpIsStatic, DynamicIotaIsStatic>(context);

  patterns->add<FuseConvolutionBatchNormalization>(context);
}

std::unique_ptr<Pass> createStablehloAggressiveSimplificationPass(
    GreedyRewriteConfig config) {
  return std::make_unique<StablehloAggressiveSimplificationPass>(config);
}

}  // namespace stablehlo
}  // namespace mlir
