// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements optional canonicalization patterns for StableHLO ops.

#include <cassert>
#include <functional>
#include <numeric>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

using llvm::SmallBitVector;

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOAGGRESSIVESIMPLIFICATIONPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {
// This is an upper limit on how many elements canonicalization patterns are
// allowed to materialize as new constants.
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

/// Binary constant folder that used a generic folder function to handle both
/// ints and floats.
template <typename Fn>
static TypedAttr foldBinaryOpIntOrFloat(TypedAttr lhs, TypedAttr rhs,
                                        Fn &&folder) {
  Attribute operands[2] = {lhs, rhs};
  Type elemTy = getElementTypeOrSelf(lhs);

  Attribute res;
  if (isa<IntegerType>(elemTy))
    res = constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(operands,
                                                                       folder);
  if (isa<FloatType>(elemTy))
    res = constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(operands,
                                                                   folder);
  if (res) return cast<TypedAttr>(res);

  return nullptr;
}

struct AddOpCanon final : OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto elemType = op.getType().getElementType();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (matchPattern(lhs, m_Zero())) {
      rewriter.replaceOp(op, rhs);
      return success();
    }

    if (matchPattern(rhs, m_AnyOf(m_Zero(), m_NegZeroFloat()))) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (isa<IntegerType>(elemType) && lhsAttr && !rhsAttr) {
      rewriter.modifyOpInPlace(op, [op, lhs, rhs] {
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    if (TypedAttr res;
        lhsAttr && rhsAttr &&
        (res = foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::plus<>{}))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

struct SubtractOpCanon final : OpRewritePattern<mlir::stablehlo::SubtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const override {
    auto elemType = op.getType().getElementType();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (isa<IntegerType>(elemType) && lhs == rhs) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }

    // Subtraction of 0.
    if (matchPattern(rhs, m_AnyOf(m_Zero(), m_PosZeroFloat()))) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    if (TypedAttr res;
        lhsAttr && rhsAttr &&
        (res = foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::minus<>{}))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

struct MulOpCanon final : OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto elemType = op.getType().getElementType();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Multiplication by 0. This fold is not trivial for floats in presence of
    // NaN values.
    if (matchPattern(lhs, m_Zero())) {
      rewriter.replaceOp(op, lhs);
      return success();
    }
    if (matchPattern(rhs, m_Zero())) {
      rewriter.replaceOp(op, rhs);
      return success();
    }

    // Multiplication by 1.
    if (matchPattern(rhs, m_One())) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (isa<IntegerType>(elemType) && lhsAttr && !rhsAttr) {
      rewriter.modifyOpInPlace(op, [op, lhs, rhs] {
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    if (TypedAttr res;
        lhsAttr && rhsAttr &&
        (res = foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::multiplies<>{}))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

static mlir::stablehlo::ComparisonDirection invertDirection(
    mlir::stablehlo::ComparisonDirection direction) {
  using mlir::stablehlo::ComparisonDirection;

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

static APInt calculateComp(mlir::stablehlo::ComparisonType kind,
                           mlir::stablehlo::ComparisonDirection direction,
                           const APInt &lhs, const APInt &rhs) {
  using mlir::stablehlo::ComparisonDirection;
  using mlir::stablehlo::ComparisonType;
  assert(llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            kind) &&
         "Not an integer comparison");

  auto asBit = [](bool value) {
    return value ? APInt::getAllOnes(1) : APInt::getZero(1);
  };

  switch (direction) {
    case ComparisonDirection::EQ:
      return asBit(lhs == rhs);
    case ComparisonDirection::NE:
      return asBit(lhs != rhs);
    case ComparisonDirection::GE:
      return asBit(kind == ComparisonType::SIGNED ? lhs.sge(rhs)
                                                  : lhs.uge(rhs));
    case ComparisonDirection::GT:
      return asBit(kind == ComparisonType::SIGNED ? lhs.sgt(rhs)
                                                  : lhs.ugt(rhs));
    case ComparisonDirection::LE:
      return asBit(kind == ComparisonType::SIGNED ? lhs.sle(rhs)
                                                  : lhs.ule(rhs));
    case ComparisonDirection::LT:
      return asBit(kind == ComparisonType::SIGNED ? lhs.slt(rhs)
                                                  : lhs.ult(rhs));
  }

  llvm_unreachable("Unhandled case");
}

struct CompareOpCanon final : OpRewritePattern<mlir::stablehlo::CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();

    // Bail out on non-integer comparison.
    // TODO: Support more comparison types.
    using mlir::stablehlo::ComparisonType;
    std::optional<ComparisonType> compType = op.getCompareType();
    if (!compType ||
        !llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            *compType)) {
      return failure();
    }

    using mlir::stablehlo::ComparisonDirection;
    ComparisonDirection direction = op.getComparisonDirection();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (lhs == rhs) {
      switch (direction) {
        case ComparisonDirection::EQ:
        case ComparisonDirection::GE:
        case ComparisonDirection::LE: {
          rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
              op, SplatElementsAttr::get(type, rewriter.getBoolAttr(true)));
          return success();
        }
        case ComparisonDirection::GT:
        case ComparisonDirection::LT:
        case ComparisonDirection::NE: {
          rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
              op, rewriter.getZeroAttr(type));
          return success();
        }
      }
      llvm_unreachable("Unhandled case");
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (lhsAttr && !rhsAttr) {
      rewriter.modifyOpInPlace(op, [&op, direction, lhs, rhs] {
        op.setComparisonDirection(invertDirection(direction));
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    if (Attribute res;
        lhsAttr && rhsAttr &&
        (res = constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
             ArrayRef<Attribute>({lhsAttr, rhsAttr}), op.getType(),
             [direction, kind = *compType](const APInt &a, const APInt &b) {
               return calculateComp(kind, direction, a, b);
             }))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

struct SelectOpCanon final : OpRewritePattern<mlir::stablehlo::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SelectOp op,
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

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(type, newValues));
    return success();
  }
};

struct CompareSelectIntoMinMax final
    : OpRewritePattern<mlir::stablehlo::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value pred = op.getPred();
    Value trueVal = op.getOnTrue();
    Value falseVal = op.getOnFalse();

    auto cmpOp = pred.getDefiningOp<mlir::stablehlo::CompareOp>();
    if (!cmpOp) return failure();

    using mlir::stablehlo::ComparisonDirection;
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
        rewriter.replaceOpWithNewOp<mlir::stablehlo::MaxOp>(op, trueVal,
                                                            falseVal);
        return success();
      }
      case ComparisonDirection::LE:
      case ComparisonDirection::LT: {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::MinOp>(op, trueVal,
                                                            falseVal);
        return success();
      }
      default: {
        return failure();
      }
    }
  }
};

struct BroadcastInDimOpCanon final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();

    TypedValue<RankedTensorType> operand = op.getOperand();
    RankedTensorType operandTy = operand.getType();

    // Fold when broadcast is a noop.
    auto dims = op.getBroadcastDimensions();
    bool isDimsIota = isIotaRange(dims);
    if (type == operandTy && isDimsIota) {
      rewriter.replaceOp(op, operand);
      return success();
    }

    // Handle splat broadcasts.
    if (SplatElementsAttr cstAttr;
        matchPattern(operand, m_Constant(&cstAttr))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     cstAttr.getSplatValue<Attribute>()));
      return success();
    }

    if (operandTy.hasStaticShape() && type.hasStaticShape() &&
        type.getNumElements() == operandTy.getNumElements()) {
      // BroadcastInDim equivalent to reshape.
      if (isDimsIota) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, type,
                                                                operand);
        return success();
      }
      // BroadcastInDim equivalent to transpose.
      if (type.getRank() == operandTy.getRank()) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::TransposeOp>(
            op, type, operand, dims);
        return success();
      }
    }

    // Eliminate redundant nested BroadcastInDim.
    if (auto definingOp =
            operand.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto newIndices =
          llvm::map_to_vector(definingOp.getBroadcastDimensions(),
                              [&dims](int64_t dim) { return dims[dim]; });
      rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
          op, type, definingOp.getOperand(), newIndices);
      return success();
    }

    return failure();
  }
};

struct ConcatenateOpCanon final
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape()) return failure();

    size_t numElems = type.getNumElements();
    if (numElems > kFoldOpEltLimit) return failure();

    // Fold concatenate when all inputs are constants.
    OperandRange inputs = op.getInputs();
    SmallVector<DenseElementsAttr> constants(inputs.size());
    for (auto [input, constant] : llvm::zip_equal(inputs, constants)) {
      if (!matchPattern(input, m_Constant(&constant))) return failure();
    }

    uint64_t dim = op.getDimension();
    ArrayRef<int64_t> shape = type.getShape();
    int64_t topSize = std::accumulate(shape.begin(), shape.begin() + dim,
                                      int64_t{1}, std::multiplies<>{});

    SmallVector<Attribute> newElems;
    newElems.reserve(numElems);

    for (int64_t i = 0; i != topSize; ++i) {
      for (ElementsAttr attr : constants) {
        size_t bottomSize = attr.getNumElements() / topSize;
        auto begin = attr.value_begin<Attribute>() + (i * bottomSize);
        newElems.append(begin, begin + bottomSize);
      }
    }

    assert(newElems.size() == numElems);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), newElems));
    return success();
  }
};

struct ConvertOpCanon final : OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this convert is a noop.
    if (op.getOperand().getType() != op.getType()) return failure();

    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

/// Does the same as PatternRewriter::replaceOpWithNewOp, but with a twist.
///
/// Sometimes, we want to replace an op with a new op and simultaneously refine
/// the result type from a dynamically-shaped type to a statically-shaped type.
/// (Search for usages of this function for examples).
//
/// Oftentimes, this works just fine because HLO is designed to accommodate
/// this kind of type refinements. But sometimes, this doesn't work - when
/// the op is used outside of the HLO dialect (e.g. in func.return). In these
/// cases, we insert a tensor.cast to smooth things out.
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
      replacementResult = rewriter.create<mlir::tensor::CastOp>(
          op->getLoc(), opResult.getType(), newOpResult);
    replacementResults.push_back(replacementResult);
  }

  rewriter.replaceOp(op, replacementResults);
  return newOp;
}

/// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
/// BroadcastInDimOp.
struct DynamicBroadcastInDimOpNotActuallyDynamic final
    : OpRewritePattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType operandType = op.getOperand().getType();
    if (!operandType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires operand static shape");

    RankedTensorType type = op.getType();
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
          op, type, op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }

    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (llvm::SmallVector<int64_t> shape;
        succeeded(hlo::matchInts(op.getOutputDimensions(), shape))) {
      refineOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
          rewriter, op, RankedTensorType::get(shape, type.getElementType()),
          op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

struct ChainedDynamicBroadcastInDimCanonicalization final
    : OpRewritePattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp bcast,
                                PatternRewriter &rewriter) const override {
    auto precedingBcast =
        bcast.getOperand()
            .getDefiningOp<mlir::stablehlo::DynamicBroadcastInDimOp>();
    if (!precedingBcast) return failure();

    // Compose broadcast dimensions.
    SmallVector<int64_t> composition;
    for (int64_t precedingDim : precedingBcast.getBroadcastDimensions()) {
      composition.push_back(bcast.getBroadcastDimensions()[precedingDim]);
    }
    auto composedBcastDims = rewriter.getDenseI64ArrayAttr(composition);

    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), precedingBcast.getOperand(),
        bcast.getOutputDimensions(), composedBcastDims);
    return success();
  }
};

// If all dimensions are known to be nonexpanding from the attribute, replace
// the dynamic broadcast with a cast.
struct DynamicBroadcastInDimAllDimsNonExpanding final
    : OpRewritePattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType type = op.getType();

    if (!op.getKnownNonexpandingDimensions() ||
        static_cast<int64_t>(op.getKnownNonexpandingDimensions()->size()) !=
            type.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "known_nonexpanding_dimensions don't cover all output dims");
    }

    auto cast = rewriter.createOrFold<tensor::CastOp>(op.getLoc(), type,
                                                      op.getOperand());
    rewriter.replaceOp(op, cast);
    return success();
  }
};

struct NoopReduceOpCanon final : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // No dimensions to reduce.
    if (op.getDimensions().empty()) {
      rewriter.replaceOp(op, op.getInputs());
      return success();
    }

    // If all returned values in the ReduceOp region exists outside the
    // region, replace the ReduceOp with those values.
    if (auto retOp = dyn_cast<mlir::stablehlo::ReturnOp>(
            op.getBody().front().getTerminator())) {
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

struct EmptyReduceOpCanon final : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
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
        bcast = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(loc, outTy,
                                                                   init, empty);
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
      bcast = rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
          loc, outTy, init, shape, empty);
    }
    rewriter.replaceOp(op, broadcasts);
    return success();
  }
};

struct UnusedResultReduceOpCanon final
    : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
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
    auto retOp = cast<mlir::stablehlo::ReturnOp>(reducerBlock.getTerminator());

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

    auto newOp = rewriter.create<ReduceOp>(
        op.getLoc(), newInputs, newInitVals,
        cast<DenseI64ArrayAttr>(op.getDimensionsAttr()), newElementTypes);
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

    rewriter.create<mlir::stablehlo::ReturnOp>(retOp.getLoc(),
                                               newReturnOperands);

    // Build new results list (unused entries will be null).
    SmallVector<Value> newResults(op.getNumResults());
    for (const auto &[i, result] : llvm::enumerate(usedResults)) {
      newResults[result.getResultNumber()] = newOp.getResult(i);
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct DynamicReshapeOpCanon final
    : OpRewritePattern<mlir::stablehlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // This is a noop when the output type is already a static shape.
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape()) return failure();

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, type,
                                                            op.getOperand());
    return success();
  }
};

struct GetTupleElementOpCanon final
    : OpRewritePattern<mlir::stablehlo::GetTupleElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GetTupleElementOp op,
                                PatternRewriter &rewriter) const override {
    auto tuple = op.getOperand().getDefiningOp<mlir::stablehlo::TupleOp>();
    if (!tuple) return failure();

    Value result = tuple.getOperand(op.getIndex());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RealOpCanon final : OpRewritePattern<mlir::stablehlo::RealOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::RealOp op,
                                PatternRewriter &rewriter) const override {
    auto complex = op.getOperand().getDefiningOp<mlir::stablehlo::ComplexOp>();
    if (!complex) return failure();

    rewriter.replaceOp(op, complex.getLhs());
    return success();
  }
};

struct ImagOpCanon final : OpRewritePattern<mlir::stablehlo::ImagOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ImagOp op,
                                PatternRewriter &rewriter) const override {
    auto complex = op.getOperand().getDefiningOp<mlir::stablehlo::ComplexOp>();
    if (!complex) return failure();

    rewriter.replaceOp(op, complex.getRhs());
    return success();
  }
};

struct GetDimensionSizeOpCanon final
    : OpRewritePattern<mlir::stablehlo::GetDimensionSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold get_dimension_size when the queried dim is statically known.
    RankedTensorType operandTy = op.getOperand().getType();

    int64_t dimSize = operandTy.getDimSize(op.getDimension());
    if (dimSize < 0) return failure();

    auto elemTy = cast<IntegerType>(op.getType().getElementType());
    IntegerAttr elemVal = rewriter.getIntegerAttr(elemTy, dimSize);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), elemVal));
    return success();
  }
};

/// Converts gather ops to slice ops in case we have a single set of constant
/// indices.
struct GatherOpCanon final : OpRewritePattern<mlir::stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GatherOp gather,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index)))
      return failure();

    mlir::stablehlo::GatherDimensionNumbersAttr dnums =
        gather.getDimensionNumbers();
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
    Value result = rewriter.create<mlir::stablehlo::SliceOp>(
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
      result = rewriter.create<mlir::stablehlo::ReshapeOp>(gather.getLoc(),
                                                           reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

struct ReshapeOpCanon final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold noop reshape.
    if (op.getType() == op.getOperand().getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    // Fold reshape of a constant.
    ElementsAttr cstAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&cstAttr))) return failure();

    if (auto splat = dyn_cast<SplatElementsAttr>(cstAttr)) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     splat.getSplatValue<Attribute>()));
      return success();
    }

    auto elements =
        llvm::to_vector_of<Attribute>(cstAttr.getValues<Attribute>());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), elements));
    return success();
  }
};

struct MergeConsecutiveReshapes final
    : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold noop reshape.
    auto operand = op.getOperand();
    if (op.getType() == operand.getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    // Fold reshape(reshape(x)).
    auto reshapeOp = operand.getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return rewriter.notifyMatchFailure(
          op, "requires defining op of operand to be Reshape");

    op.setOperand(reshapeOp->getOperand(0));
    return success();
  }
};

struct TransposeIsReshape final
    : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto permutation = op.getPermutation();

    if (isIotaRange(permutation)) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    RankedTensorType inputTy = input.getType();
    if (!inputTy.hasStaticShape() || !op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "requires input and output to be of a statically-shaped ranked "
          "tensor type");

    SmallVector<int64_t> permValues(permutation);
    SmallVector<int64_t> nonZeroPerms;
    nonZeroPerms.reserve(permValues.size());
    for (auto idx : permValues) {
      auto sz = inputTy.getDimSize(idx);
      if (sz != 1) nonZeroPerms.push_back(idx);
    }

    for (int i = 1, s = nonZeroPerms.size(); i < s; ++i)
      if (nonZeroPerms[i - 1] > nonZeroPerms[i])
        return rewriter.notifyMatchFailure(op, "memory layout change");

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, op.getType(),
                                                            input);
    return success();
  }
};

/// Check if a `t` is a tensor with zero extents.
static std::optional<RankedTensorType> isZeroExtent(Type t) {
  auto type = dyn_cast<RankedTensorType>(t);
  if (type && type.hasStaticShape() && type.getNumElements() == 0) return type;
  return std::nullopt;
}

// Replace instances of zero extent tensors with empty tensors of the same
// type.
struct ZeroExtentTensorCanon final : RewritePattern {
  ZeroExtentTensorCanon(MLIRContext *context, PatternBenefit benefit)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (!isa_and_present<mlir::stablehlo::StablehloDialect>(op->getDialect()))
      return rewriter.notifyMatchFailure(op, "not stablehlo");

    // If the result is a zero-extent tensor, replace the whole op with an empty
    // tensor.
    bool didUpdate = false;
    for (auto result : op->getResults()) {
      auto resultType = isZeroExtent(result.getType());
      if (!resultType || result.use_empty()) continue;
      rewriter.replaceAllUsesWith(result, rewriter.create<tensor::EmptyOp>(
                                              loc, resultType->getShape(),
                                              resultType->getElementType()));
      didUpdate = true;
    }

    // If one of the operands is a zero-extent tensor, replace the operand with
    // an empty tensor.
    for (OpOperand &operand : op->getOpOperands()) {
      auto operandType = isZeroExtent(operand.get().getType());
      if (!operandType || operand.get().getDefiningOp<tensor::EmptyOp>())
        continue;
      Operation *owner = operand.getOwner();
      int operandNum = operand.getOperandNumber();
      auto emptyTensorOp = rewriter.create<tensor::EmptyOp>(
          loc, operandType->getShape(), operandType->getElementType());
      rewriter.modifyOpInPlace(
          owner, [&]() { owner->setOperand(operandNum, emptyTensorOp); });
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

    if (!isa<mlir::stablehlo::ReshapeOp>(definingOp) &&
        !isa<mlir::stablehlo::TransposeOp>(definingOp) &&
        !isa<mlir::stablehlo::BroadcastOp>(definingOp))
      return rewriter.notifyMatchFailure(
          op, "defining operation of unexpected type");

    // Only reorder if the defining op has no other uses.
    if (!llvm::hasSingleElement(definingOp->getResult(0).getUses()))
      return rewriter.notifyMatchFailure(op, "operation has more than one use");

    Value input = definingOp->getOperand(0);
    Value result = op->getResult(0);
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

struct StablehloAggressiveSimplificationPass final
    : impl::StablehloAggressiveSimplificationPassBase<
          StablehloAggressiveSimplificationPass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet patterns_(context);
    populateStablehloCanonicalizationPatterns(context, &patterns_);
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

void populateStablehloCanonicalizationPatterns(MLIRContext *context,
                                               RewritePatternSet *patterns,
                                               PatternBenefit benefit) {
  patterns->add<
      // Arithmetic ops.
      AddOpCanon, SubtractOpCanon, MulOpCanon, CompareOpCanon, SelectOpCanon,
      CompareSelectIntoMinMax,
      // Complex ops.
      RealOpCanon, ImagOpCanon,
      // Query ops.
      GetDimensionSizeOpCanon, GetTupleElementOpCanon,
      // Broadcast ops.
      BroadcastInDimOpCanon, DynamicBroadcastInDimOpNotActuallyDynamic,
      ChainedDynamicBroadcastInDimCanonicalization,
      DynamicBroadcastInDimAllDimsNonExpanding,
      // Reduce op.
      NoopReduceOpCanon, EmptyReduceOpCanon, UnusedResultReduceOpCanon,
      // Shape manipulation(-ish) ops.
      ConcatenateOpCanon, ConvertOpCanon, DynamicReshapeOpCanon, GatherOpCanon,
      ReshapeOpCanon, MergeConsecutiveReshapes, TransposeIsReshape,
      // Types.
      ZeroExtentTensorCanon>(context, benefit);
  patterns->add<ReorderElementwiseAndShapeOp>(context);
}

}  // namespace stablehlo
}  // namespace mlir
