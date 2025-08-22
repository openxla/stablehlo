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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"

#define DEBUG_TYPE "stablehlo-optimization"

namespace mlir::stablehlo {

#define GEN_PASS_DEF_STABLEHLOAGGRESSIVEFOLDERPASS
#include "stablehlo/transforms/optimization/Passes.h.inc"

namespace {

static constexpr StablehloAggressiveFolderPassOptions kDefaultOptions;

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
  return APSInt(
      {/*numBits=*/numBits, value, /*isSigned=*/false, /*implicitTrunc=*/true},
      /*isUnsigned=*/isUnsigned);
}

APFloat getAPFloat(
    Type type, double value,
    llvm::RoundingMode roundingMode = llvm::RoundingMode::NearestTiesToEven) {
  auto floatType = dyn_cast<FloatType>(type);
  if (!floatType) llvm::report_fatal_error("expected float type");

  APFloat result(value);
  bool losesInfo = false;
  result.convert(floatType.getFloatSemantics(), roundingMode, &losesInfo);
  return result;
}

LogicalResult validateStaticShapeResult(PatternRewriter& rewriter,
                                        Operation* op, ShapedType resultType) {
  if (!resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "unable to fold dynamically shaped result type to constant");
  return success();
}

/// Unary constant folder that uses a generic folder function to handle both
/// ints and floats.
template <typename Fn, typename IntResultType = IntegerAttr,
          typename FloatResultType = FloatAttr>
TypedAttr foldUnaryOpIntOrFloat(Type resultType, TypedAttr operand,
                                Fn&& folder) {
  Type elemTy = getElementTypeOrSelf(operand);

  Attribute res;
  if (isa<IntegerType>(elemTy))
    res = constFoldUnaryOp<IntegerAttr, IntegerAttr::ValueType, void,
                           IntResultType>(operand, folder);
  if (isa<FloatType>(elemTy))
    res = constFoldUnaryOp<FloatAttr, FloatAttr::ValueType, void,
                           FloatResultType>(operand, folder);
  if (res) return cast<TypedAttr>(res);

  return nullptr;
}

/// Unary constant folder that uses a generic folder function to handle both
/// ints and floats.
template <typename Fn, typename IntResultType = IntegerAttr,
          typename FloatResultType = FloatAttr>
FailureOr<TypedAttr> foldUnaryOpIntOrFloat(PatternRewriter& rewriter,
                                           Operation* op, Fn&& folder) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected unary op");

  TypedAttr attr;
  matchPattern(op->getOperand(0), m_Constant(&attr));

  if (!attr) return rewriter.notifyMatchFailure(op, "operand not constants");

  TypedAttr res = foldUnaryOpIntOrFloat<Fn, IntResultType, FloatResultType>(
      op->getResultTypes()[0], attr, std::forward<Fn>(folder));
  if (!res) return rewriter.notifyMatchFailure(op, "folding failed");

  return res;
}

/// Binary constant folder that uses a generic folder function to handle both
/// ints and floats.
template <typename Fn, typename IntResultType = IntegerAttr,
          typename FloatResultType = FloatAttr>
TypedAttr foldBinaryOpIntOrFloat(Type resultType, TypedAttr lhs, TypedAttr rhs,
                                 Fn&& folder) {
  Attribute operands[2] = {lhs, rhs};
  Type elemTy = getElementTypeOrSelf(lhs);

  Attribute res;
  if (isa<IntegerType>(elemTy))
    res = constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void,
                            IntResultType>(operands, resultType, folder);
  if (isa<FloatType>(elemTy))
    res = constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void,
                            FloatResultType>(operands, resultType, folder);
  if (res) return cast<TypedAttr>(res);

  return nullptr;
}

/// Binary constant folder that uses a generic folder function to handle both
/// ints and floats.
template <typename Fn, typename IntResultType = IntegerAttr,
          typename FloatResultType = FloatAttr>
FailureOr<TypedAttr> foldBinaryOpIntOrFloat(PatternRewriter& rewriter,
                                            Operation* op, Fn&& folder) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected binary op");

  TypedAttr lhsAttr, rhsAttr;
  matchPattern(op->getOperand(0), m_Constant(&lhsAttr));
  matchPattern(op->getOperand(1), m_Constant(&rhsAttr));

  if (!lhsAttr || !rhsAttr)
    return rewriter.notifyMatchFailure(op, "lhs & rhs operands not constants");

  TypedAttr res = foldBinaryOpIntOrFloat<Fn, IntResultType, FloatResultType>(
      op->getResultTypes()[0], lhsAttr, rhsAttr, std::forward<Fn>(folder));
  if (!res) return rewriter.notifyMatchFailure(op, "folding failed");

  return res;
}

template <class AttrElementT, class TargetAttrElementT, class CalculationT,
          typename OpType>
LogicalResult foldConvertHelper(PatternRewriter& rewriter, OpType op,
                                DenseIntOrFPElementsAttr elements, Type resType,
                                CalculationT&& calculate) {
  auto result = constFoldCastOp<AttrElementT, TargetAttrElementT,
                                typename AttrElementT::ValueType,
                                typename TargetAttrElementT::ValueType, void>(
      elements, resType, calculate);

  if (!result) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
      diag << "cast of " << elements.getElementType() << " to " << resType
           << " failed";
    });
  }

  rewriter.replaceOpWithNewOp<ConstantOp>(op, result);
  return success();
}

template <typename OpType>
LogicalResult foldConvert(PatternRewriter& rewriter, OpType op,
                          DenseIntOrFPElementsAttr elements,
                          RankedTensorType resultType) {
  auto oldType = getElementTypeOrSelf(elements);
  auto newType = getElementTypeOrSelf(resultType);
  size_t newBitWidth = newType.getIntOrFloatBitWidth();

  bool isOldTypeUnsigned =
      oldType.isSignlessInteger(1) || oldType.isUnsignedInteger();
  bool isNewTypeUnsigned =
      newType.isSignlessInteger(1) || newType.isUnsignedInteger();
  bool isNewTypeBoolean = newType.isSignlessInteger(1);

  if (isa<FloatType>(oldType)) {
    if (auto newFloatType = dyn_cast<FloatType>(newType)) {
      // Float -> Float
      const auto& targetSemantics = newFloatType.getFloatSemantics();
      return foldConvertHelper<FloatAttr, FloatAttr>(
          rewriter, op, elements, resultType,
          [&targetSemantics](const APFloat& operand, bool& castStatus) {
            bool losesInfo;
            APFloat newValue = operand;
            castStatus = APFloat::opInvalidOp !=
                         newValue.convert(targetSemantics,
                                          llvm::RoundingMode::NearestTiesToEven,
                                          &losesInfo);
            return newValue;
          });
    }

    // Float -> Boolean
    if (isNewTypeBoolean) {
      return foldConvertHelper<FloatAttr, IntegerAttr>(
          rewriter, op, elements, resultType,
          [newBitWidth](const APFloat& operand, bool& /*castStatus*/) {
            APInt resVal(1, operand.isZero() ? 0 : 1);
            return resVal.sextOrTrunc(newBitWidth);
          });
    }

    // Float -> Int
    return foldConvertHelper<FloatAttr, IntegerAttr>(
        rewriter, op, elements, resultType,
        [&newBitWidth, &isNewTypeUnsigned](const APFloat& operand,
                                           bool& castStatus) {
          APSInt api(newBitWidth, isNewTypeUnsigned);
          if (operand.isInfinity() || operand.isNegZero()) {
            castStatus = false;
            return api;
          }
          bool ignored;
          castStatus =
              APFloat::opInvalidOp !=
              operand.convertToInteger(api, APFloat::rmTowardZero, &ignored);
          return api;
        });
  }

  if (auto newFloatType = dyn_cast<FloatType>(newType)) {
    // Int -> Float
    return foldConvertHelper<IntegerAttr, FloatAttr>(
        rewriter, op, elements, resultType,
        [&newFloatType, &isOldTypeUnsigned](const APInt& operand,
                                            bool& /*castStatus*/) {
          APFloat apf(newFloatType.getFloatSemantics(),
                      APInt::getZero(newFloatType.getWidth()));
          apf.convertFromAPInt(operand, !isOldTypeUnsigned,
                               APFloat::rmNearestTiesToEven);
          return apf;
        });
  }

  // Int -> Boolean
  if (isNewTypeBoolean) {
    return foldConvertHelper<IntegerAttr, IntegerAttr>(
        rewriter, op, elements, resultType,
        [newBitWidth](const APInt& operand, bool& /*castStatus*/) {
          APInt resVal(1, operand.isZero() ? 0 : 1);
          return resVal.sextOrTrunc(newBitWidth);
        });
  }

  // Int -> Int
  return foldConvertHelper<IntegerAttr, IntegerAttr>(
      rewriter, op, elements, resultType,
      [&newBitWidth, &isOldTypeUnsigned](const APInt& operand,
                                         bool& /*castStatus*/) {
        return APSInt(operand, isOldTypeUnsigned).extOrTrunc(newBitWidth);
      });
}

bool hasAnyDeclaredSideEffects(Operation* op) {
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Return true if the op explicitly declares any memory effects of its own.
    if (!memInterface.hasNoEffect()) return true;
    // The op has no direct memory effects. Return false if it has no recursive
    // memory effects, either.
    if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return false;
  }

  // The op doesn't declare any side effects of its own, but its regions could
  // still contain ops that do declare side effects. Recursively check them.
  for (Region& region : op->getRegions()) {
    for (Operation& nestedOp : region.getOps()) {
      if (hasAnyDeclaredSideEffects(&nestedOp)) return true;
    }
  }
  return false;
}

template <typename OpType>
struct FoldOpRewritePattern : OpRewritePattern<OpType> {
  FoldOpRewritePattern(MLIRContext* context,
                       const StablehloAggressiveFolderPassOptions& options,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<OpType>(context, benefit, generatedNames),
        options(options) {}

  // Prevent `options` from binding to a temporary.
  FoldOpRewritePattern(MLIRContext* context,
                       StablehloAggressiveFolderPassOptions&& options,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {}) = delete;

  const StablehloAggressiveFolderPassOptions& options;

  LogicalResult validateElementCountForFold(PatternRewriter& rewriter,
                                            Operation* op,
                                            ShapedType resultType) const {
    size_t numElems = resultType.getNumElements();
    if (numElems > static_cast<size_t>(options.foldOpElementLimit))
      return rewriter.notifyMatchFailure(
          op,
          "too many elements, fold "
          "limit is " +
              std::to_string(options.foldOpElementLimit));
    return success();
  }
};

// A base class to use for patterns that may be used for integer shape math,
// but also may be used for general folding of floats.
template <typename OpType>
struct ShapeOpRewritePattern : public FoldOpRewritePattern<OpType> {
  using FoldOpRewritePattern<OpType>::FoldOpRewritePattern;
  using FoldOpRewritePattern<OpType>::matchAndRewrite;
  using FoldOpRewritePattern<OpType>::options;

  LogicalResult validateShapeFoldDtype(PatternRewriter& rewriter, OpType op,
                                       ShapedType resultType) const {
    if (resultType.getElementType().isInteger()) return success();
    if (options.optimizeFloat && isa<FloatType>(resultType.getElementType()))
      return success();
    return rewriter.notifyMatchFailure(op, "skipping fold of shape op dtype");
  }
};

struct FoldAddOpPattern final
    : public ShapeOpRewritePattern<mlir::stablehlo::AddOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter& rewriter) const override {
    if (failed(validateShapeFoldDtype(rewriter, op, op.getType())))
      return failure();

    auto res = foldBinaryOpIntOrFloat(rewriter, op, std::plus<>{});
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }
};

struct FoldAndOpPattern : public ShapeOpRewritePattern<AndOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(AndOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    auto resultElementType = resultType.getElementType();
    FailureOr<TypedAttr> result;

    if (resultElementType.isInteger(/*width=*/1)) {
      result = foldBinaryOpIntOrFloat(rewriter, op, FoldLogicalAnd{});
    } else if (resultElementType.isInteger()) {
      result = foldBinaryOpIntOrFloat(rewriter, op, FoldBitwiseAnd{});
    } else {
      return rewriter.notifyMatchFailure(op, "Expected integral element type.");
    }

    if (failed(result)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op,
                                                             result.value());
    return success();
  }

  struct FoldLogicalAnd {
    APInt operator()(APInt lhs, APInt rhs) const {
      return APInt(lhs.getBitWidth(), !lhs.isZero() && !rhs.isZero());
    }
    std::optional<APFloat> operator()(APFloat lhs, APFloat rhs) const {
      return std::nullopt;
    }
  };

  struct FoldBitwiseAnd {
    APInt operator()(APInt lhs, APInt rhs) const { return lhs & rhs; }

    std::optional<APFloat> operator()(APFloat lhs, APFloat rhs) const {
      return std::nullopt;
    }
  };
};

// Pattern: broadcast_in_dim(splat, _) -> constant(splat)
struct FoldBroadcastInDimOpSplatPattern
    : public ShapeOpRewritePattern<BroadcastInDimOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    SplatElementsAttr cstAttr;
    matchPattern(op.getOperand(), m_Constant(&cstAttr));
    if (!cstAttr) return rewriter.notifyMatchFailure(op, "operand not splat");

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, SplatElementsAttr::get(op.getType(),
                                   cstAttr.getSplatValue<Attribute>()));
    return success();
  }
};

struct FoldCompareOpPattern : public ShapeOpRewritePattern<CompareOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    auto res = foldBinaryOpIntOrFloat<FoldCompare, IntegerAttr, IntegerAttr>(
        rewriter, op,
        FoldCompare(op.getComparisonDirection(), op.getCompareType()));
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }

  struct FoldCompare {
    FoldCompare(ComparisonDirection direction,
                std::optional<ComparisonType> kind)
        : direction(direction), kind(kind) {}
    ComparisonDirection direction;
    std::optional<ComparisonType> kind;

    APInt operator()(APFloat lhs, APFloat rhs) {
      bool result = false;
      switch (direction) {
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
      return APInt(/*bitwidth=*/1, result);
    }
    APInt operator()(APInt lhs, APInt rhs) {
      bool result = false;
      switch (direction) {
        case ComparisonDirection::EQ:
          result = lhs == rhs;
          break;
        case ComparisonDirection::NE:
          result = lhs != rhs;
          break;
        case ComparisonDirection::GE:
          result = kind == ComparisonType::SIGNED ? lhs.sge(rhs) : lhs.uge(rhs);
          break;
        case ComparisonDirection::GT:
          result = kind == ComparisonType::SIGNED ? lhs.sgt(rhs) : lhs.ugt(rhs);
          break;
        case ComparisonDirection::LE:
          result = kind == ComparisonType::SIGNED ? lhs.sle(rhs) : lhs.ule(rhs);
          break;
        case ComparisonDirection::LT:
          result = kind == ComparisonType::SIGNED ? lhs.slt(rhs) : lhs.ult(rhs);
          break;
      }
      return APInt(/*bitwidth=*/1, result);
    }
  };
};

//////////////////////////////////
// CaseOp
/////////////////////////////////

class InlineCaseOpWithConstantBranchIndex
    : public FoldOpRewritePattern<CaseOp> {
 public:
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(CaseOp op,
                                PatternRewriter& rewriter) const override {
    // Fail to match dead `case` ops. Dead-code elimination should already erase
    // such ops whenever it's safe to do so; if we find a dead `case` op that
    // can't be erased, we need to signal match failure or else the pattern will
    // be reapplied ad infinitum.
    if (op->use_empty())
      return rewriter.notifyMatchFailure(op, "The case op's result is unused.");

    Value branchIndexArgument = op.getIndex();
    SplatElementsAttr indexAttr;
    if (!matchPattern(branchIndexArgument, m_Constant(&indexAttr)))
      return rewriter.notifyMatchFailure(op, "Branch index is not a constant.");

    int64_t selectedBranchIndex =
        indexAttr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
    // If the branch index is OOB, the last branch is executed by default:
    // https://openxla.org/stablehlo/spec#case
    if (selectedBranchIndex < 0 || selectedBranchIndex >= op.getNumRegions())
      selectedBranchIndex = op.getNumRegions() - 1;

    Region& region = op.getRegion(selectedBranchIndex);
    assert(region.hasOneBlock());
    Block* blockToInline = &region.front();
    ValueRange blockArgs = {};
    Operation* terminator = blockToInline->getTerminator();
    ValueRange results = terminator->getOperands();

    // TODO: Add support for complex, quantized, and token return types.
    // Currently, this pattern only supports int and float return types. We'll
    // need a more general equivalent of `getZeroAttr` to support other types.
    SmallVector<TypedAttr> placeholderAttrs;
    for (auto result : op.getResults()) {
      TypedAttr placeholderAttr = rewriter.getZeroAttr(result.getType());
      if (!placeholderAttr)
        return rewriter.notifyMatchFailure(
            op,
            "The case op's return type isn't currently supported by this "
            "optimization pattern.");
      placeholderAttrs.push_back(placeholderAttr);
    }

    // Inline the active branch of the `case` op.
    rewriter.inlineBlockBefore(blockToInline, op, blockArgs);
    rewriter.replaceAllOpUsesWith(op, results);
    rewriter.eraseOp(terminator);

    // Make sure the now-dead `case` op is still syntactically valid in case it
    // can't be safely deleted (e.g. due to side effects). Specifically, we left
    // one region of the `case` op empty when we inlined that block; it expects
    // a block with a terminator op, so we need to make it return something.
    Block& noopBlock = region.emplaceBlock();
    SmallVector<Value> placeholderResults;
    rewriter.setInsertionPointToEnd(&noopBlock);
    for (auto placeholderAttr : placeholderAttrs) {
      placeholderResults.push_back(
          rewriter.create<ConstantOp>(region.getLoc(), placeholderAttr));
    }
    rewriter.create<stablehlo::ReturnOp>(region.getLoc(), placeholderResults);

    return success();
  }
};

//////////////////////////////////
// ConcatenateOp
/////////////////////////////////

struct FoldConcatenateOpPattern final
    : ShapeOpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    RankedTensorType type = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, type)) ||
        failed(validateShapeFoldDtype(rewriter, op, type)) ||
        failed(validateElementCountForFold(rewriter, op, type)))
      return failure();

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
    size_t numElems = type.getNumElements();
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

  int64_t foldOpElementLimit;
};

struct FoldConvertOpPattern : public ShapeOpRewritePattern<ConvertOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto operand = op.getOperand();
    RankedTensorType resultType = op.getType();

    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    auto operandElemType = getElementTypeOrSelf(operand.getType());
    auto resultElemType = getElementTypeOrSelf(resultType);
    if (!options.optimizeFloat &&
        (isa<FloatType>(operandElemType) || isa<FloatType>(resultElemType)))
      return rewriter.notifyMatchFailure(op, "skipping fold of float convert");

    DenseIntOrFPElementsAttr elements;
    if (!matchPattern(operand, m_Constant(&elements)))
      return rewriter.notifyMatchFailure(
          op, "expected constant integer or float operand");

    return foldConvert(rewriter, op, elements, resultType);
  }
};

struct FoldDivOpPattern : public ShapeOpRewritePattern<DivOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    bool isUnsignedInt = resultType.getElementType().isUnsignedInteger();
    auto res = foldBinaryOpIntOrFloat(rewriter, op, FoldDivide(isUnsignedInt));
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }

  struct FoldDivide {
    FoldDivide(bool isUnsignedInt)
        : foldIntFn(isUnsignedInt ? foldUint : foldSint) {}
    std::function<APInt(APInt, APInt)> foldIntFn;

    APFloat operator()(APFloat lhs, APFloat rhs) { return lhs / rhs; }
    APInt operator()(APInt lhs, APInt rhs) { return foldIntFn(lhs, rhs); }
    static APInt foldUint(APInt lhs, APInt rhs) { return lhs.udiv(rhs); }
    static APInt foldSint(APInt lhs, APInt rhs) { return lhs.sdiv(rhs); }
  };
};

struct FoldGetDimensionSizeOpPattern
    : public ShapeOpRewritePattern<GetDimensionSizeOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(GetDimensionSizeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
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

/////
// Max/Min/Clamp
/////

struct FoldMax {
  FoldMax(bool isUnsignedInt)
      : foldIntFn(isUnsignedInt ? foldUint : foldSint) {}
  std::function<APInt(APInt, APInt)> foldIntFn;

  APFloat operator()(APFloat lhs, APFloat rhs) {
    return lhs >= rhs ? lhs : rhs;
  }
  APInt operator()(APInt lhs, APInt rhs) { return foldIntFn(lhs, rhs); }
  static APInt foldUint(APInt lhs, APInt rhs) {
    return lhs.uge(rhs) ? lhs : rhs;
  }
  static APInt foldSint(APInt lhs, APInt rhs) {
    return lhs.sge(rhs) ? lhs : rhs;
  }
};

struct FoldMin {
  FoldMin(bool isUnsignedInt)
      : foldIntFn(isUnsignedInt ? foldUint : foldSint) {}
  std::function<APInt(APInt, APInt)> foldIntFn;

  APFloat operator()(APFloat lhs, APFloat rhs) {
    return lhs <= rhs ? lhs : rhs;
  }
  APInt operator()(APInt lhs, APInt rhs) { return foldIntFn(lhs, rhs); }
  static APInt foldUint(APInt lhs, APInt rhs) {
    return lhs.ule(rhs) ? lhs : rhs;
  }
  static APInt foldSint(APInt lhs, APInt rhs) {
    return lhs.sle(rhs) ? lhs : rhs;
  }
};

struct FoldMaxOpPattern : public ShapeOpRewritePattern<MaxOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    auto resultElementType = resultType.getElementType();
    if (failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    bool isUnsignedIntOrBool = resultElementType.isUnsignedInteger() ||
                               resultElementType.isInteger(/*width=*/1);
    auto res =
        foldBinaryOpIntOrFloat(rewriter, op, FoldMax(isUnsignedIntOrBool));
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }
};

struct FoldMinOpPattern : public ShapeOpRewritePattern<MinOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(MinOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    auto resultElementType = resultType.getElementType();
    if (failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    bool isUnsignedIntOrBool = resultElementType.isUnsignedInteger() ||
                               resultElementType.isInteger(/*width=*/1);
    auto res =
        foldBinaryOpIntOrFloat(rewriter, op, FoldMin(isUnsignedIntOrBool));
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }
};

// Clamp is folded using Min and Max folders.
struct FoldClampOpPattern : public ShapeOpRewritePattern<ClampOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(ClampOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    TypedAttr minAttr, operandAttr, maxAttr;
    matchPattern(op.getMin(), m_Constant(&minAttr));
    matchPattern(op.getOperand(), m_Constant(&operandAttr));
    matchPattern(op.getMax(), m_Constant(&maxAttr));

    if (!minAttr || !operandAttr || !maxAttr)
      return rewriter.notifyMatchFailure(op, "operands not constant");

    // Fold clamp using:
    //   res = max(min, operand)
    //   res = min(max, res)
    bool isUnsignedInt = resultType.getElementType().isUnsignedInteger();
    auto res = foldBinaryOpIntOrFloat(resultType, minAttr, operandAttr,
                                      FoldMax(isUnsignedInt));
    res = foldBinaryOpIntOrFloat(resultType, maxAttr, res,
                                 FoldMin(isUnsignedInt));
    if (!res) return rewriter.notifyMatchFailure(op, "failed to fold clamp");
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
    return success();
  }
};

struct FoldMulOpPattern final : ShapeOpRewritePattern<mlir::stablehlo::MulOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter& rewriter) const override {
    if (failed(validateShapeFoldDtype(rewriter, op, op.getType())))
      return failure();

    auto res = foldBinaryOpIntOrFloat(rewriter, op, std::multiplies<>{});
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }
};

struct FoldOrOpPattern : public ShapeOpRewritePattern<OrOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(OrOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    auto resultElementType = resultType.getElementType();
    FailureOr<TypedAttr> result;

    if (resultElementType.isInteger(/*width=*/1)) {
      result = foldBinaryOpIntOrFloat(rewriter, op, FoldLogicalOr{});
    } else if (resultElementType.isInteger()) {
      result = foldBinaryOpIntOrFloat(rewriter, op, FoldBitwiseOr{});
    } else {
      return rewriter.notifyMatchFailure(op, "Expected integral element type.");
    }

    if (failed(result)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op,
                                                             result.value());
    return success();
  }

  struct FoldLogicalOr {
    APInt operator()(APInt lhs, APInt rhs) const {
      return APInt(lhs.getBitWidth(), !lhs.isZero() || !rhs.isZero());
    }
    std::optional<APFloat> operator()(APFloat lhs, APFloat rhs) const {
      return std::nullopt;
    }
  };

  struct FoldBitwiseOr {
    APInt operator()(APInt lhs, APInt rhs) const { return lhs | rhs; }

    std::optional<APFloat> operator()(APFloat lhs, APFloat rhs) const {
      return std::nullopt;
    }
  };
};

struct FoldRemOpPattern : public ShapeOpRewritePattern<RemOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(RemOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    bool isUnsignedInt = resultType.getElementType().isUnsignedInteger();
    auto res = foldBinaryOpIntOrFloat(rewriter, op, FoldRem(isUnsignedInt));
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }

  struct FoldRem {
    FoldRem(bool isUnsignedInt)
        : foldIntFn(isUnsignedInt ? foldUint : foldSint) {}
    std::function<APInt(APInt, APInt)> foldIntFn;

    std::optional<APFloat> operator()(APFloat lhs, APFloat rhs) {
      // `APFloat::mod` requires both operands to have identical semantics.
      if (&lhs.getSemantics() != &rhs.getSemantics()) return std::nullopt;

      lhs.mod(rhs);  // This modifies `lhs` in place.
      return lhs;    // `lhs` now holds the result.
    }
    APInt operator()(APInt lhs, APInt rhs) { return foldIntFn(lhs, rhs); }
    static APInt foldUint(APInt lhs, APInt rhs) { return lhs.urem(rhs); }
    static APInt foldSint(APInt lhs, APInt rhs) { return lhs.srem(rhs); }
  };
};

// Pattern: reshape(cst, shape) -> cst
struct FoldReshapeOpPattern : public ShapeOpRewritePattern<ReshapeOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    DenseIntOrFPElementsAttr attr;
    if (!matchPattern(op.getOperand(), m_Constant(&attr)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");
    rewriter.replaceOpWithNewOp<ConstantOp>(op, attr.reshape(resultType));
    return success();
  }
};

struct FoldSelectOpPattern : public ShapeOpRewritePattern<SelectOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    DenseIntElementsAttr predAttr;
    DenseElementsAttr onTrueAttr, onFalseAttr;
    matchPattern(op.getPred(), m_Constant(&predAttr));
    matchPattern(op.getOnTrue(), m_Constant(&onTrueAttr));
    matchPattern(op.getOnFalse(), m_Constant(&onFalseAttr));
    if (!predAttr || !onTrueAttr || !onFalseAttr)
      return rewriter.notifyMatchFailure(op, "expected constant operands");

    // Optimization, handle splat predicate
    if (isa<SplatElementsAttr>(predAttr)) {
      auto pred = predAttr.getSplatValue<APInt>();
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, pred.isZero() ? onFalseAttr : onTrueAttr);
      return success();
    }

    // TODO: Enable float folding.
    if (op.getType().getElementType().isFloat())
      return rewriter.notifyMatchFailure(op, "float select not supported yet");

    // Fall back to verbose folding
    if (failed(validateElementCountForFold(rewriter, op, resultType)))
      return failure();

    SmallVector<APInt> result;
    for (auto [predEl, onTrueEl, onFalseEl] :
         llvm::zip(predAttr.getValues<APInt>(), onTrueAttr.getValues<APInt>(),
                   onFalseAttr.getValues<APInt>())) {
      result.push_back(!predEl.isZero() ? onTrueEl : onFalseEl);
    }
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(resultType, result));

    return success();
  }

  struct FoldSelect {
    std::optional<APFloat> operator()(APFloat pred, APFloat onTrue,
                                      APFloat onFalse) {
      return std::nullopt;
    }

    APInt operator()(APInt pred, APInt onTrue, APInt onFalse) {
      return pred != 0 ? onTrue : onFalse;
    }
  };
};

// Pattern: set_dim_size(op, size, dim) -> op [op[dim] == size]
struct FoldSetDimensionSizeOpPattern
    : public ShapeOpRewritePattern<SetDimensionSizeOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(SetDimensionSizeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    // No need to verify static shape or dtype here since we aren't evaluating
    // dtype, just folding set_dim_size ops with no semantic meaning.

    SplatElementsAttr cstSplatAttr;
    matchPattern(op.getSize(), m_Constant(&cstSplatAttr));
    if (!cstSplatAttr)
      return rewriter.notifyMatchFailure(op, "size operand not constant");

    // Compare to the dimension size, not the bound size.
    // We can't fold and change shapes, but we do want to fold meaningless
    // set_dim_size ops that use a constant to set a static dimension size
    // like `set_dim_size(X, 2, dim=0) : (tensor<2xf32>) -> tensor<2xf32>`
    if (cstSplatAttr.getSplatValue<APInt>() !=
        resultType.getDimSize(op.getDimension()))
      return rewriter.notifyMatchFailure(op,
                                         "dim size does not match result type");

    rewriter.replaceAllOpUsesWith(op, op.getOperand());
    return success();
  }
};

struct FoldSignOpPattern : public ShapeOpRewritePattern<SignOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(SignOp op,
                                PatternRewriter& rewriter) const override {
    if (failed(validateShapeFoldDtype(rewriter, op, op.getType())))
      return failure();

    auto elementType = op.getType().getElementType();
    auto res = foldUnaryOpIntOrFloat(rewriter, op, FoldSign(elementType));
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }

  struct FoldSign {
    FoldSign(Type elementType) : elementType(elementType) {}
    Type elementType;
    double result;
    APFloat operator()(APFloat operand) {
      if (operand.isNegative())
        result = -1.0;
      else if (operand.isZero())
        result = 0.0;
      else
        result = 1.0;
      return getAPFloat(elementType, result);
    }

    APInt operator()(APInt operand) {
      // SignOp only supports signed integers.
      APSInt signedInt = getAPSInt(elementType, operand.getSExtValue());
      int64_t result;
      if (signedInt.isNegative())
        result = -1;
      else if (signedInt.isZero())
        result = 0;
      else
        result = 1;
      return getAPSInt(elementType, result);
    }
  };
};

template <typename RangeType>
DenseElementsAttr sliceType(SliceOp& op, const RangeType& data) {
  using ElementType = std::decay_t<decltype(*std::begin(data))>;

  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType resultType = op.getResult().getType();

  const auto dimOffsets = computeStrides(operandType.getShape());
  auto startIndices = op.getStartIndices();
  auto limitIndices = op.getLimitIndices();
  auto strides = op.getStrides();

  const SmallVector<int64_t> startIndex(startIndices);
  const SmallVector<int64_t> endIndex(limitIndices);

  SmallVector<ElementType> result;
  result.reserve(resultType.getNumElements());

  SmallVector<int64_t> srcIndex(startIndex);
  for (int64_t i = 0; i < resultType.getNumElements(); ++i) {
    auto srcLinearIndex = linearize(srcIndex, dimOffsets);
    result.push_back(data[srcLinearIndex]);
    for (int64_t dim = srcIndex.size() - 1; dim >= 0; --dim) {
      srcIndex[dim] += strides[dim];
      if (srcIndex[dim] >= endIndex[dim])
        srcIndex[dim] = startIndex[dim];
      else
        break;
    }
  }

  return DenseElementsAttr::get(op.getResult().getType(),
                                ArrayRef<ElementType>(result));
}

struct FoldSliceOpPattern : public ShapeOpRewritePattern<SliceOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    auto operand = op.getOperand();
    RankedTensorType operandType = operand.getType();
    if (!operandType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected operand with static ranked tensor type");

    ElementsAttr els;
    if (!matchPattern(operand, m_Constant(&els)))
      return rewriter.notifyMatchFailure(
          op, "expected constant integer or float operand");

    DenseElementsAttr resAttr;
    if (auto data = els.tryGetValues<APInt>())
      resAttr = sliceType(op, *data);
    else if (auto data = els.tryGetValues<APFloat>())
      resAttr = sliceType(op, *data);
    else
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unsupported element type");

    rewriter.replaceOpWithNewOp<ConstantOp>(op, resAttr);
    return success();
  }
};

struct FoldSubtractOpPattern final
    : ShapeOpRewritePattern<mlir::stablehlo::SubtractOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter& rewriter) const override {
    if (failed(validateShapeFoldDtype(rewriter, op, op.getType())))
      return failure();

    auto res = foldBinaryOpIntOrFloat(rewriter, op, std::minus<>{});
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }
};

struct FoldSqrtOpPattern
    : public FoldOpRewritePattern<mlir::stablehlo::SqrtOp> {
  using FoldOpRewritePattern<mlir::stablehlo::SqrtOp>::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SqrtOp op,
                                PatternRewriter& rewriter) const final {
    auto res = foldUnaryOpIntOrFloat(rewriter, op, FoldSqrt());
    if (failed(res)) return failure();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res.value());
    return success();
  }

  struct FoldSqrt {
    std::optional<APFloat> operator()(APFloat operand) {
      if (operand.getSizeInBits(operand.getSemantics()) == 64)
        return APFloat(std::sqrt(operand.convertToDouble()));

      if (operand.getSizeInBits(operand.getSemantics()) == 32)
        return APFloat(sqrtf(operand.convertToFloat()));
      return std::nullopt;
    }

    // TODO: Enable int folding.
    std::optional<APInt> operator()(APInt operand) { return std::nullopt; }
  };
};

struct FoldIotaOpPattern : public FoldOpRewritePattern<IotaOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "FoldIotaOpPattern folding: " << op << '\n');
    auto resultType = cast<RankedTensorType>(op.getType());
    if (failed(validateElementCountForFold(rewriter, op, resultType)))
      return failure();

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

  int64_t foldOpElementLimit;
};

template <typename RangeType>
DenseElementsAttr transposeType(TransposeOp& op, const RangeType& data) {
  using ElementType = std::decay_t<decltype(*std::begin(data))>;

  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType resultType = op.getResult().getType();

  const auto operandStrides = computeStrides(operandType.getShape());
  const auto resultStrides = computeStrides(resultType.getShape());
  const auto inversePermutation = invertPermutationVector(op.getPermutation());

  SmallVector<ElementType> result;
  result.reserve(resultType.getNumElements());

  for (int64_t i = 0; i < resultType.getNumElements(); ++i) {
    auto dstDimIndex = delinearize(i, resultStrides);
    auto srcDimIndex = applyPermutation(dstDimIndex, inversePermutation);
    auto srcLinearIndex = linearize(srcDimIndex, operandStrides);
    result.push_back(data[srcLinearIndex]);
  }

  return DenseElementsAttr::get(resultType, ArrayRef<ElementType>(result));
}

// transpose(constant) => constant with permuted dimensions
// This covers ranked tensor types with 0 dimensions(zero elements) and 0
// rank(scalar), as well as splat values.
struct FoldTransposeOpPattern : public FoldOpRewritePattern<TransposeOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)))
      return failure();

    ElementsAttr els;
    if (!matchPattern(op.getOperand(), m_Constant(&els)))
      return rewriter.notifyMatchFailure(
          op, "expected constant integer or float operand");

    // TODO: Does this expand splat values? Should we special case splats?
    DenseElementsAttr resAttr;
    if (auto data = els.tryGetValues<APInt>())
      resAttr = transposeType(op, *data);
    else if (auto data = els.tryGetValues<APFloat>())
      resAttr = transposeType(op, *data);
    else
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unsupported element type");

    rewriter.replaceOpWithNewOp<ConstantOp>(op, resAttr);
    return success();
  }
};

// TODO: Consider moving this into aggressive simplifications.
struct LowerBoolSplatConstantsIntoReduceOpRegion
    : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& body = op.getBody().front();

    if (body.getOperations().size() != 2)
      return rewriter.notifyMatchFailure(op, "Incompatible op count in body.");
    if (!isa<AndOp, OrOp>(body.front()))
      return rewriter.notifyMatchFailure(op, "Only match AND and OR ops.");

    SmallVector<DenseElementsAttr, 4> bodyArgConstantAttrs;

    for (auto [inputValue, bodyArg] :
         llvm::zip_equal(op.getOperands(), body.getArguments())) {
      auto inputConstantOp = inputValue.getDefiningOp<ConstantOp>();
      if (!inputConstantOp)
        return rewriter.notifyMatchFailure(op, "Input must be a constant.");

      auto inputConstantAttr =
          dyn_cast_or_null<DenseElementsAttr>(inputConstantOp.getValue());
      if (!inputConstantAttr)
        return rewriter.notifyMatchFailure(op,
                                           "Input must be a splat constant.");

      auto bodyArgShapedType = dyn_cast<ShapedType>(bodyArg.getType());
      if (!bodyArgShapedType)
        return rewriter.notifyMatchFailure(
            op, "Could not get the shape of the body argument.");

      bodyArgConstantAttrs.push_back(DenseElementsAttr::get(
          bodyArgShapedType, inputConstantAttr.getSplatValue<Attribute>()));
    }

    for (BlockArgument bodyArg : body.getArguments()) {
      rewriter.replaceAllUsesWith(
          bodyArg, rewriter.create<ConstantOp>(
                       body.front().getLoc(), bodyArg.getType(),
                       bodyArgConstantAttrs[bodyArg.getArgNumber()]));
    }

    return success();
  }
};

struct FoldReduceOpReducingZeroDims : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    // Fail to match if the reduce op operates on any dimensions.
    if (!op.getDimensions().empty())
      return rewriter.notifyMatchFailure(
          op, "The reduce op reduces a nonzero number of dimensions.");

    // Check that input and output types match.
    for (auto [in, out] : llvm::zip_equal(op.getInputs(), op.getResults())) {
      if (in.getType() != out.getType())
        return rewriter.notifyMatchFailure(
            op, "Input and output types do not match.");
    }

    rewriter.replaceOp(op, op.getInputs());
    return success();
  }
};

struct FoldReduceOpToConstantInitializer
    : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& body = op.getBody().front();
    if (body.getOperations().size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "Body must contain exactly one op.");

    auto returnOp = dyn_cast<ReturnOp>(body.back());
    if (!returnOp)
      return rewriter.notifyMatchFailure(op, "Body must end with a return op.");

    SmallVector<DenseElementsAttr> resultAttrs;
    for (auto [bodyResult, opResult] :
         llvm::zip_equal(returnOp.getResults(), op.getResults())) {
      auto* sourceOfBlockResult = bodyResult.getDefiningOp();
      if (!sourceOfBlockResult ||
          !sourceOfBlockResult->hasTrait<OpTrait::ConstantLike>())
        return rewriter.notifyMatchFailure(op,
                                           "Body result must be a constant.");

      DenseElementsAttr constantAttr;
      if (!matchPattern(sourceOfBlockResult, m_Constant(&constantAttr)))
        return rewriter.notifyMatchFailure(
            op, "Could not extract constant attribute from body result.");

      auto resultShapedType = dyn_cast<ShapedType>(opResult.getType());
      if (!resultShapedType)
        return rewriter.notifyMatchFailure(
            op, "Could not get the shape of the reduce op's result.");

      resultAttrs.push_back(DenseElementsAttr::get(
          resultShapedType, {constantAttr.getSplatValue<Attribute>()}));
    }

    SmallVector<Value> resultValues;
    for (auto resultAttr : resultAttrs) {
      resultValues.push_back(rewriter.create<ConstantOp>(
          op.getLoc(), resultAttr.getType(), resultAttr));
    }

    rewriter.replaceOp(op, resultValues);
    return success();
  }
};

struct FoldReduceOpWithRedundantResults
    : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& body = op.getBody().front();
    auto returnOp = dyn_cast<ReturnOp>(body.back());
    if (!returnOp)
      return rewriter.notifyMatchFailure(op, "Body must end with a return op.");

    Region* returnOpParentRegion = returnOp->getParentRegion();

    for (auto [reduceOpResult, returnOpResult] :
         llvm::zip_equal(op.getResults(), returnOp.getResults())) {
      if (returnOpResult.getParentRegion() == returnOpParentRegion ||
          returnOpResult.getType() != reduceOpResult.getType()) {
        return rewriter.notifyMatchFailure(
            op, "The reduce op's result isn't redundant.");
      }
    }
    rewriter.replaceOp(op, returnOp.getResults());
    return success();
  }
};

// Return success if the while condition is always false.
LogicalResult whileCondIsFalse(WhileOp op, PatternRewriter& rewriter) {
  auto condReturnOp = dyn_cast<ReturnOp>(op.getCond().front().back());
  if (!condReturnOp)
    return rewriter.notifyMatchFailure(
        op, "Condition region is missing a return statement.");

  DenseIntElementsAttr condValue;
  if (!matchPattern(condReturnOp.getOperand(0), m_Constant(&condValue)))
    return rewriter.notifyMatchFailure(
        op, "Condition block does not return a constant.");
  if (condValue.getSplatValue<BoolAttr>().getValue())
    return rewriter.notifyMatchFailure(
        op, "Condition value is not a splat of the bool `false`.");

  return success();
}

// Pattern: while(operands, cond=false) -> operands
// Replace a while loop's uses with its operands if the condition is always
// false.
struct FoldWhileOpPattern : public FoldOpRewritePattern<WhileOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    // It is, unfortunately, possible for code to depend on the very existence
    // of a side effect even if that side effect is unreachable. We'd ideally
    // like to fix this, but that's not as simple as it sounds. For now, we need
    // to make sure we don't DCE code with side effects in case something else
    // depends on it.
    if (op->use_empty() && !isOpTriviallyDead(op))
      return rewriter.notifyMatchFailure(
          op, "Keeping dead while op due to known or potential side effects.");

    // Check if the while condition is always false, i.e. body never executed.
    if (failed(whileCondIsFalse(op, rewriter))) return failure();

    // Replace uses of the op's result, but don't remove the op itself; let
    // dedicated DCE logic handle that step if appropriate. (This is because of
    // the aforementioned issue where ops with side effects might need to remain
    // in the IR even if unreachable.)
    rewriter.replaceAllOpUsesWith(op, op.getOperand());
    return success();
  }
};

// Pattern: while(cond=false) { call @fn } -> ()
// `CallOp`s are considered to have side effects by default, but we may want to
// DCE a `WhileOp` that will never execute its body regardless of `CallOp` side
// effects. This is because dead while loops containing large kernels can cause
// massive increases in program size that can in turn cause OOMs in compilation.
//
// TODO: Add a pre-processing pass to annotate call ops that are safe to
// DCE and a pattern to remove dead call ops; then we won't need this check.
struct FoldWhileOpIfDeadAndPresumedPure : public FoldOpRewritePattern<WhileOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    if (!options.assumeNoUndeclaredSideEffects)
      return rewriter.notifyMatchFailure(
          op,
          "Pattern skipped: only applicable if `assumeNoUndeclaredSideEffects` "
          "is enabled.");

    // Only delete while ops that are dead and cannot be executed. We should
    // preserve any `WhileOp` that may be executed, e.g. in case its body
    // contains debug prints.
    if (!op->use_empty() || failed(whileCondIsFalse(op, rewriter)))
      return rewriter.notifyMatchFailure(op, "The while body may run.");

    if (hasAnyDeclaredSideEffects(op))
      return rewriter.notifyMatchFailure(
          op,
          "The op, or another op within its region, explicitly declares side "
          "effects.");

    // Neither this op nor any in its regions have any declared side effects,
    // `assumeNoUndeclaredSideEffects` is true, and the op's result is unused.
    // Erase the op.
    rewriter.eraseOp(op);
    return success();
  }
};

struct StablehloAggressiveFolderPass
    : public impl::StablehloAggressiveFolderPassBase<
          StablehloAggressiveFolderPass> {
  explicit StablehloAggressiveFolderPass(
      StablehloAggressiveFolderPassOptions options,
      GreedyRewriteConfig rewriteConfig = {})
      : StablehloAggressiveFolderPassBase(options),
        rewriteConfig(rewriteConfig) {}

  explicit StablehloAggressiveFolderPass()
      : StablehloAggressiveFolderPassBase() {}

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    StablehloAggressiveFolderPassOptions options{
        /*assumeNoUndeclaredSideEffects=*/assumeNoUndeclaredSideEffects,
        /*foldOpElementLimit=*/foldOpElementLimit,
        /*optimizeFloat=*/optimizeFloat,
    };

    populateStablehloAggressiveFolderPatterns(context, &patterns, options);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     rewriteConfig)))
      signalPassFailure();
  }

 private:
  GreedyRewriteConfig rewriteConfig;
};

}  // namespace

void populateStablehloAggressiveFolderPatterns(
    MLIRContext* context, RewritePatternSet* patterns,
    const StablehloAggressiveFolderPassOptions& options,
    PatternBenefit benefit) {
  populateStablehloShapeFolderPatterns(context, patterns, options, benefit);

  patterns->add<FoldIotaOpPattern,                    //
                FoldReduceOpReducingZeroDims,         //
                FoldReduceOpToConstantInitializer,    //
                FoldReduceOpWithRedundantResults,     //
                FoldSqrtOpPattern,                    //
                FoldTransposeOpPattern,               //
                FoldWhileOpIfDeadAndPresumedPure,     //
                FoldWhileOpPattern,                   //
                InlineCaseOpWithConstantBranchIndex,  //
                LowerBoolSplatConstantsIntoReduceOpRegion>(context, options,
                                                           benefit);
}

class StablehloTargetIndependentOptimizationPass {
 private:
  StablehloTargetIndependentOptimizationPassOptions options;
};

class StablehloIndependentOptimizationPass {
 private:
  StablehloTargetIndependentOptimizationPassOptions options;
};

void populateStablehloShapeFolderPatterns(
    MLIRContext* context, RewritePatternSet* patterns,
    const StablehloAggressiveFolderPassOptions& options,
    PatternBenefit benefit) {
  patterns->add<FoldAddOpPattern>(context, options, benefit);
  patterns->add<FoldAndOpPattern>(context, options, benefit);
  patterns->add<FoldBroadcastInDimOpSplatPattern>(context, options, benefit);
  patterns->add<FoldClampOpPattern>(context, options, benefit);
  patterns->add<FoldCompareOpPattern>(context, options, benefit);
  patterns->add<FoldConcatenateOpPattern>(context, options, benefit);
  patterns->add<FoldConvertOpPattern>(context, options, benefit);
  patterns->add<FoldDivOpPattern>(context, options, benefit);
  patterns->add<FoldGetDimensionSizeOpPattern>(context, options, benefit);
  patterns->add<FoldMaxOpPattern>(context, options, benefit);
  patterns->add<FoldMinOpPattern>(context, options, benefit);
  patterns->add<FoldMulOpPattern>(context, options, benefit);
  patterns->add<FoldOrOpPattern>(context, options, benefit);
  patterns->add<FoldRemOpPattern>(context, options, benefit);
  patterns->add<FoldReshapeOpPattern>(context, options, benefit);
  patterns->add<FoldSelectOpPattern>(context, options, benefit);
  patterns->add<FoldSetDimensionSizeOpPattern>(context, options, benefit);
  patterns->add<FoldSignOpPattern>(context, options, benefit);
  patterns->add<FoldSliceOpPattern>(context, options, benefit);
  patterns->add<FoldSubtractOpPattern>(context, options, benefit);
}

void populateStablehloShapeFolderPatterns(MLIRContext* context,
                                          RewritePatternSet* patterns,
                                          PatternBenefit benefit) {
  populateStablehloShapeFolderPatterns(context, patterns, kDefaultOptions,
                                       benefit);
}

std::unique_ptr<::mlir::Pass> createStablehloAggressiveFolderPass(
    StablehloAggressiveFolderPassOptions options,
    GreedyRewriteConfig rewriteConfig) {
  return std::make_unique<StablehloAggressiveFolderPass>(options,
                                                         rewriteConfig);
}

}  // namespace mlir::stablehlo
