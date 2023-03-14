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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOREFINESHAPESPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

// The patterns below implement partial evaluation of shape computations which
// is a critical part of implementing type refinement for ops like
// dynamic_broadcast_in_dim, dynamic_iota and dynamic_reshape whose shape
// depends on the value of their shape operands.

template <typename T>
LogicalResult matchInts(Value value, SmallVector<T>& result) {
  DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return failure();
  for (auto element : attr.getValues<APInt>()) {
    if constexpr (std::is_same<T, int64_t>::value) {
      result.push_back(element.getSExtValue());
    } else {
      result.push_back(element);
    }
  }
  return success();
}

template <typename OpType, typename FuncType>
LogicalResult evalUnary(PatternRewriter& rewriter, OpType op, FuncType fn) {
  if (op->getNumOperands() != 1)
    llvm::report_fatal_error("expected one operand");

  auto resultType = op.getResult().getType().template cast<ShapedType>();
  if (!resultType.hasRank() || !resultType.getElementType().isIntOrIndex())
    return rewriter.notifyMatchFailure(
        op, "expected integer or index result tensor type");

  SmallVector<APInt> operand, result;
  if (failed(matchInts(op.getOperand(), operand)))
    return rewriter.notifyMatchFailure(op, "expected constant operand");
  for (const auto& operandEl : operand) {
    result.push_back(fn(operandEl));
  }

  rewriter.replaceOpWithNewOp<ConstantOp>(
      op, DenseIntElementsAttr::get(resultType, result));
  return success();
}

template <typename OpType, typename FuncType>
LogicalResult evalBinary(PatternRewriter& rewriter, OpType op, FuncType fn) {
  if (op->getNumOperands() != 2)
    llvm::report_fatal_error("expected two operands");

  auto resultType = op.getResult().getType().template cast<ShapedType>();
  if (!resultType.hasRank() || !resultType.getElementType().isIntOrIndex())
    return rewriter.notifyMatchFailure(
        op, "expected integer or index result tensor type");

  SmallVector<APInt> lhs, rhs, result;
  if (failed(matchInts(op.getLhs(), lhs)) ||
      failed(matchInts(op.getRhs(), rhs)))
    return rewriter.notifyMatchFailure(op, "expected constant operands");
  for (auto [lhsEl, rhsEl] : llvm::zip(lhs, rhs)) {
    result.push_back(fn(lhsEl, rhsEl));
  }

  rewriter.replaceOpWithNewOp<ConstantOp>(
      op, DenseIntElementsAttr::get(resultType, result));
  return success();
}

struct EvalAddOpPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    return evalBinary(rewriter, op,
                      [&](APInt lhs, APInt rhs) { return lhs + rhs; });
  }
};

struct EvalAndOpPattern : public OpRewritePattern<AndOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AndOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType().cast<ShapedType>();
    if (!resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "expected boolean element type");

    return evalBinary(rewriter, op, [&](APInt lhsInt, APInt rhsInt) {
      auto result = lhsInt != 0 && rhsInt != 0;
      return APInt(/*numBits=*/1, result);
    });
  }
};

struct EvalBroadcastInDimOpPattern : public OpRewritePattern<BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!operandType || operandType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "expected 0-dimensional type");

    SmallVector<APInt> operand;
    if (failed(matchInts(op.getOperand(), operand)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");
    auto scalar = operand[0];

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(op.getResult().getType(), scalar));
    return success();
  }
};

struct EvalCompareOpPattern : public OpRewritePattern<CompareOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType().cast<ShapedType>();
    auto isResultUnsigned = resultType.getElementType().isUnsignedInteger();
    return evalBinary(rewriter, op, [&](APInt lhs, APInt rhs) {
      bool result;
      switch (op.getComparisonDirection()) {
        case ComparisonDirection::EQ:
          result = lhs == rhs;
          break;
        case ComparisonDirection::NE:
          result = lhs != rhs;
          break;
        case ComparisonDirection::GE:
          result = isResultUnsigned ? lhs.uge(rhs) : lhs.sge(rhs);
          break;
        case ComparisonDirection::GT:
          result = isResultUnsigned ? lhs.ugt(rhs) : lhs.sgt(rhs);
          break;
        case ComparisonDirection::LE:
          result = isResultUnsigned ? lhs.ule(rhs) : lhs.sle(rhs);
          break;
        case ComparisonDirection::LT:
          result = isResultUnsigned ? lhs.ult(rhs) : lhs.slt(rhs);
          break;
      }
      return APInt(/*numBits=*/1, result);
    });
  }
};

struct EvalConcatenateOpPattern : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType || op.getDimension() != 0)
      return rewriter.notifyMatchFailure(op, "expected dimension = 0");

    SmallVector<APInt> result;
    for (Value operand : op->getOperands()) {
      if (failed(matchInts(operand, result)))
        return rewriter.notifyMatchFailure(op, "expected constant operands");
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(resultType, result));
    return success();
  }
};

struct EvalConvertOpPattern : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType().cast<ShapedType>();
    auto isOperandUnsigned = operandType.getElementType().isUnsignedInteger();
    auto resultType = op.getResult().getType().cast<ShapedType>();
    if (!resultType.getElementType().isIntOrIndex())
      return rewriter.notifyMatchFailure(op,
                                         "expected integer result tensor type");
    auto resultBitwidth = resultType.getElementType().getIntOrFloatBitWidth();
    return evalUnary(rewriter, op, [&](APInt operand) {
      return APSInt(operand, isOperandUnsigned).extOrTrunc(resultBitwidth);
    });
  }
};

struct EvalDivOpPattern : public OpRewritePattern<DivOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType().cast<ShapedType>();
    auto isResultUnsigned = resultType.getElementType().isUnsignedInteger();
    return evalBinary(rewriter, op, [&](APInt lhs, APInt rhs) {
      return isResultUnsigned ? lhs.udiv(rhs) : lhs.sdiv(rhs);
    });
  }
};

struct EvalGetDimensionSizeOpPattern
    : public OpRewritePattern<GetDimensionSizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetDimensionSizeOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!operandType)
      return rewriter.notifyMatchFailure(op, "expected ranked operand");
    if (operandType.isDynamicDim(op.getDimension()))
      return rewriter.notifyMatchFailure(op, "expected static dimension");

    auto resultType = op.getResult().getType().cast<RankedTensorType>();
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
    auto resultType = op.getResult().getType().cast<ShapedType>();
    auto isResultUnsigned = resultType.getElementType().isUnsignedInteger();
    return evalBinary(rewriter, op, [&](APInt lhs, APInt rhs) {
      return isResultUnsigned ? llvm::APIntOps::umax(lhs, rhs)
                              : llvm::APIntOps::smax(lhs, rhs);
    });
  }
};

struct EvalMulOpPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    return evalBinary(rewriter, op,
                      [&](APInt lhs, APInt rhs) { return lhs * rhs; });
  }
};

struct EvalRemOpPattern : public OpRewritePattern<RemOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RemOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType().cast<ShapedType>();
    auto isResultUnsigned = resultType.getElementType().isUnsignedInteger();
    return evalBinary(rewriter, op, [&](APInt lhs, APInt rhs) {
      return isResultUnsigned ? lhs.urem(rhs) : lhs.srem(rhs);
    });
  }
};

struct EvalReshapeOpPattern : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    DenseIntElementsAttr attr;
    if (!matchPattern(op.getOperand(), m_Constant(&attr)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");
    rewriter.replaceOpWithNewOp<ConstantOp>(op, attr.reshape(op.getType()));
    return success();
  }
};

struct EvalSelectOpPattern : public OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<APInt> pred, onTrue, onFalse;
    if (failed(matchInts(op.getPred(), pred)) ||
        failed(matchInts(op.getOnTrue(), onTrue)) ||
        failed(matchInts(op.getOnFalse(), onFalse)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");

    SmallVector<APInt> result;
    for (auto [predEl, onTrueEl, onFalseEl] :
         llvm::zip(pred, onTrue, onFalse)) {
      result.push_back(predEl != 0 ? onTrueEl : onFalseEl);
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(op.getResult().getType(), result));
    return success();
  }
};

struct EvalSignOpPattern : public OpRewritePattern<SignOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SignOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType().cast<ShapedType>();
    if (!resultType.getElementType().isIntOrIndex())
      return rewriter.notifyMatchFailure(op,
                                         "expected integer result tensor type");
    auto resultBitwidth = resultType.getElementType().getIntOrFloatBitWidth();
    return evalUnary(rewriter, op, [&](APInt operand) {
      int64_t result;
      if (operand.isNegative())
        result = -1;
      else if (operand.isZero())
        result = 0;
      else
        result = 1;
      return APInt(resultBitwidth, result, /*isSigned=*/true);
    });
  }
};

struct EvalSliceOpPattern : public OpRewritePattern<SliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!resultType || resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-dimensional type");

    SmallVector<int64_t> operand;
    if (failed(matchInts(op.getOperand(), operand)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");

    int64_t start = op.getStartIndices().getValues<int64_t>()[0];
    int64_t limit = op.getLimitIndices().getValues<int64_t>()[0];
    int64_t stride = op.getStrides().getValues<int64_t>()[0];
    SmallVector<int64_t> result;
    for (auto i = start; i < limit; i += stride) {
      result.push_back(operand[i]);
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(resultType, result));
    return success();
  }
};

struct EvalSubtractOpPattern : public OpRewritePattern<SubtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter& rewriter) const override {
    return evalBinary(rewriter, op,
                      [&](APInt lhs, APInt rhs) { return lhs - rhs; });
  }
};

// The patterns below implement shape refinement of individual ops.
// In a nutshell, they use the upstream type inference infrastructure and a
// StableHLO-specific extension to refine return types based on potentially
// refined operands.

// Refines the values using the given types.
// Tricky implementation details:
//   1) Need to support partial shape refinements, e.g. if just a single
//      dimension size out of an entire tensor type got refined. This is done
//      via inferMostSpecificType.
//   2) Need to signal propagation of the refined shapes across the
//      StableHLO program. Different callers of this function have different
//      propagation needs, so this function doesn't signal anything on its own
//      and leaves that to the callers.
LogicalResult refineValues(PatternRewriter& rewriter, Operation* op,
                           ValueRange values, TypeRange types) {
  if (values.size() != types.size())
    return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
      diag << "refineValues failed for " << types << ": expected "
           << values.size() << " types, got " << types.size();
    });

  // Check whether `types` contain any new information with respect to existing
  // return types. Even if just a single dimension size out of an entire tensor
  // type got updated, using `inferMostSpecificType` ensures that we don't
  // miss that.
  bool needsRefinement = false;
  SmallVector<Type> refinedTypes;
  for (auto it : llvm::zip(values.getTypes(), types)) {
    // Cannot use structured bindings to simplify this because capturing
    // structured bindings in a lambda is a C++ 20 extension.
    auto currentReturnType = std::get<0>(it);
    auto refinement = std::get<1>(it);
    auto refinedType = hlo::inferMostSpecificType(
        /*location=*/{}, {currentReturnType, refinement});
    if (failed(refinedType))
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "inferMostSpecificType failed for " << currentReturnType
             << " and " << refinement;
      });
    refinedTypes.push_back(*refinedType);
    needsRefinement |= (currentReturnType != *refinedType);
  }
  if (!needsRefinement)
    return rewriter.notifyMatchFailure(op, "doesn't need refinement");

  for (auto it : llvm::zip(values, refinedTypes)) {
    // Cannot use structured bindings to simplify this because capturing
    // structured bindings in a lambda is a C++ 20 extension.
    auto value = std::get<0>(it);
    auto refinedType = std::get<1>(it);
    if (value.getType() == refinedType) continue;

    // Check whether the users of this value are ready for the type of the
    // value to be refined.
    for (Operation* user : value.getUsers()) {
      // CHLO and StableHLO ops are designed to support type refinements of
      // their operands and results. Any operand type in these ops can change
      // within what's supported by `inferMostSpecificType` without breaking
      // verification of the op.
      if (isa<chlo::ChloDialect, StablehloDialect>(user->getDialect()))
        continue;

      // Simply changing operand type of `func.return` won't work because
      // that won't update the FunctionType of the enclosing `func.func`.
      // Nonetheless, we still want to support these ops because they are widely
      // used in StableHLO programs (although the plan of record is to replace
      // `func.return` ops in StableHLO programs with `stablehlo.return`:
      // https://github.com/openxla/stablehlo/issues/425).
      if (isa<func::ReturnOp>(user)) continue;

      // Unlike in TensorFlow's type inference pass, here we work only with
      // allowlisted ops to focus our support on well-defined semantics of
      // StableHLO programs.
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "unsupported refinement: tried to refine " << value.getType()
             << " to " << refinedType << " for user " << user;
      });
    }

    // Happy path: simply call setType here because most of our users are
    // fine with that.
    auto unrefinedType = value.getType();
    value.setType(refinedType);

    // Special case: for `func.return`, guard the refinement with a
    // `tensor.cast` and leave propagation of the refined return type to a
    // dedicated pattern.
    auto isFuncReturn = [](OpOperand& use) -> bool {
      return isa<func::ReturnOp>(use.getOwner());
    };
    if (llvm::none_of(value.getUses(), isFuncReturn)) continue;
    rewriter.setInsertionPointAfter(op);
    auto castToUnrefinedType =
        rewriter.create<tensor::CastOp>(op->getLoc(), unrefinedType, value);
    value.replaceUsesWithIf(castToUnrefinedType, isFuncReturn);
  }

  return success();
}

// Refines the return types of the given operation using the given types.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
LogicalResult refineReturnTypes(PatternRewriter& rewriter, Operation* op,
                                ArrayRef<Type> types) {
  if (failed(refineValues(rewriter, op, op->getResults(), types)))
    return failure();

  // This `replaceOpWithIf` call doesn't actually change the IR, but
  // it does ask the rewriter to visit all the users of this op. There is no
  // upstream API to achieve this directly, but if it's introduced in the
  // future, we could use it here.
  rewriter.replaceOpWithIf(op, op->getResults(),
                           [](OpOperand& use) { return false; });
  return success();
}

// Refines the return types of the given operation using the given types.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
LogicalResult refineReturnTypes(PatternRewriter& rewriter, Operation* op,
                                ArrayRef<ShapedTypeComponents> refinements) {
  if (op->getNumResults() != refinements.size())
    return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
      diag << "refineReturnTypes failed: expected " << op->getNumResults()
           << " refinements, got " << refinements.size();
    });

  SmallVector<Type> refinedTypes;
  for (auto [currentType, refinement] :
       llvm::zip(op->getResultTypes(), refinements)) {
    auto refinedElementType =
        refinement.getElementType()
            ? refinement.getElementType()
            : currentType.cast<ShapedType>().getElementType();
    if (refinement.hasRank()) {
      auto refinedShape = refinement.getDims();
      auto refinedEncoding = refinement.getAttribute();
      refinedTypes.push_back(RankedTensorType::get(
          refinedShape, refinedElementType, refinedEncoding));
    } else {
      refinedTypes.push_back(UnrankedTensorType::get(refinedElementType));
    }
  }

  return refineReturnTypes(rewriter, op, refinedTypes);
}

// Refines the return type of the given operation using the given shape.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
template <typename OpType>
LogicalResult refineReturnShape(PatternRewriter& rewriter, OpType op,
                                ArrayRef<int64_t> shape) {
  return refineReturnTypes(rewriter, op, ShapedTypeComponents(shape));
}

// Refines the return type of the given operation using the given shape.
// This function also signals PatternRewriter that it needs to visit all the
// users of this op if any updates to its results have happened during execution
// of the function.
template <typename OpType>
LogicalResult refineReturnShape(PatternRewriter& rewriter, OpType op,
                                Value shapeValue) {
  // At the moment, we only support refining return types using fully static
  // shape values which serves the current use cases well.
  // Support for partially static shape values is left for future work.
  SmallVector<int64_t> shape;
  if (failed(matchInts(shapeValue, shape)))
    return rewriter.notifyMatchFailure(op, "expected constant output shape");
  return refineReturnShape(rewriter, op, shape);
}

struct RefineAllGatherOpPattern : public OpRewritePattern<AllGatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AllGatherOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType().cast<ShapedType>();
    if (!operandType.hasRank())
      return rewriter.notifyMatchFailure(op, "expected ranked operand type");

    // This represents the cross_replica_and_partition process grouping strategy
    // that requires num_partitions to compute shardCount. Since we don't know
    // num_partitions at this point, we error out.
    if (op.getChannelHandle() && !op.getUseGlobalDeviceIds())
      return rewriter.notifyMatchFailure(op, "unsupported strategy");
    DenseIntElementsAttr replicaGroups = op.getReplicaGroups();
    auto shardCount = replicaGroups.getType().getDimSize(1);

    SmallVector<int64_t> refinement(operandType.getShape());
    if (!operandType.isDynamicDim(op.getAllGatherDim()))
      refinement[op.getAllGatherDim()] *= shardCount;
    return refineReturnShape(rewriter, op, refinement);
  }
};

struct RefineBitcastConvertOpPattern
    : public OpRewritePattern<BitcastConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BitcastConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType().cast<ShapedType>();
    if (!operandType.hasRank())
      return rewriter.notifyMatchFailure(op, "expected ranked operand type");

    // If bitwidths of the operand and the result are different, then
    // operand and result shapes have different ranks.
    // This complicates the logic quite a bit and is not needed to pass the
    // current tests, so we leave this for future work.
    auto resultType = op.getResult().getType().cast<ShapedType>();
    auto getBitWidthFn = [](ShapedType type) {
      auto elementType = type.getElementType();
      if (auto complexType = elementType.dyn_cast<ComplexType>())
        return complexType.getElementType().getIntOrFloatBitWidth();
      return elementType.getIntOrFloatBitWidth();
    };

    if (getBitWidthFn(operandType) != getBitWidthFn(resultType))
      return rewriter.notifyMatchFailure(op, "unsupported bitwidth");

    return refineReturnShape(rewriter, op, operandType.getShape());
  }
};

struct RefineConvertOpPattern : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferConvertOp(
            /*location=*/{}, op.getOperand(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvertOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineConvolutionOpPattern : public OpRewritePattern<ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferConvolutionOp(
            /*location=*/{}, op.getLhs().getType(), op.getRhs().getType(),
            op.getWindowStrides(), op.getPadding(), op.getLhsDilation(),
            op.getRhsDilation(), op.getWindowReversal(),
            op.getDimensionNumbers().getInputBatchDimension(),
            op.getDimensionNumbers().getInputFeatureDimension(),
            op.getDimensionNumbers().getInputSpatialDimensions(),
            op.getDimensionNumbers().getKernelInputFeatureDimension(),
            op.getDimensionNumbers().getKernelOutputFeatureDimension(),
            op.getDimensionNumbers().getKernelSpatialDimensions(),
            op.getDimensionNumbers().getOutputBatchDimension(),
            op.getDimensionNumbers().getOutputFeatureDimension(),
            op.getDimensionNumbers().getOutputSpatialDimensions(),
            op.getFeatureGroupCount(), op.getBatchGroupCount(),
            op.getPrecisionConfig(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvolutionOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineCustomCallOpPattern : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    auto operandIndicesAttr = op->getAttr("indices_of_shape_operands")
                                  .dyn_cast_or_null<DenseIntElementsAttr>();
    if (!operandIndicesAttr)
      return rewriter.notifyMatchFailure(
          op, "expected an indices_of_shape_operands attribute");

    SmallVector<ShapedTypeComponents> refinements;
    for (auto operandIndexElt : operandIndicesAttr.getValues<APInt>()) {
      int64_t operandIndex = operandIndexElt.getSExtValue();
      if (operandIndex < 0 || operandIndex >= op->getNumOperands())
        return rewriter.notifyMatchFailure(op, "expected valid operand index");
      SmallVector<int64_t> refinement;
      if (failed(matchInts(op->getOperand(operandIndex), refinement)))
        return rewriter.notifyMatchFailure(op, "expected constant operand");
      if (llvm::any_of(refinement, [&](int64_t x) { return x < 0; }))
        return rewriter.notifyMatchFailure(op, "expected non-negative sizes");
      refinements.emplace_back(refinement);
    }
    return refineReturnTypes(rewriter, op, refinements);
  }
};

struct RefineDotGeneralOpPattern : public OpRewritePattern<DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferDotGeneralOp(
            /*location=*/{}, op.getLhs().getType(), op.getRhs().getType(),
            op.getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
            op.getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
            op.getDotDimensionNumbersAttr().getLhsContractingDimensions(),
            op.getDotDimensionNumbersAttr().getRhsContractingDimensions(),
            op.getPrecisionConfig(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferDotGeneralOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineDynamicBroadcastInDimOpPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getOutputDimensions());
  }
};

struct RefineDynamicConvOpPattern : public OpRewritePattern<DynamicConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicConvOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> padding;
    if (failed(matchInts(op.getDPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected constant d_padding");
    if (op.getPadding().has_value())
      return rewriter.notifyMatchFailure(op, "expected empty padding");
    auto paddingType = RankedTensorType::get(
        op.getDPadding().getType().getShape(), rewriter.getIntegerType(64));
    auto paddingAttr = DenseIntElementsAttr::get(paddingType, padding);

    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferConvolutionOp(
            /*location=*/{}, op.getLhs().getType(), op.getRhs().getType(),
            op.getWindowStrides(), paddingAttr, op.getLhsDilation(),
            op.getRhsDilation(), op.getWindowReversal(),
            op.getDimensionNumbers().getInputBatchDimension(),
            op.getDimensionNumbers().getInputFeatureDimension(),
            op.getDimensionNumbers().getInputSpatialDimensions(),
            op.getDimensionNumbers().getKernelInputFeatureDimension(),
            op.getDimensionNumbers().getKernelOutputFeatureDimension(),
            op.getDimensionNumbers().getKernelSpatialDimensions(),
            op.getDimensionNumbers().getOutputBatchDimension(),
            op.getDimensionNumbers().getOutputFeatureDimension(),
            op.getDimensionNumbers().getOutputSpatialDimensions(),
            op.getFeatureGroupCount(), op.getBatchGroupCount(),
            op.getPrecisionConfig(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvolutionOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineDynamicIotaOpPattern : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicIotaOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getOutputShape());
  }
};

struct RefineDynamicPadOpPattern : public OpRewritePattern<DynamicPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicPadOp op,
                                PatternRewriter& rewriter) const override {
    // At the moment, we only support refining return types using fully static
    // shape values which serves the current use cases well.
    // Support for partially static shape values is left for future work.
    SmallVector<int64_t> edgePaddingLow, edgePaddingHigh, interiorPadding;
    if (failed(matchInts(op.getEdgePaddingLow(), edgePaddingLow)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant edge_padding_low");
    if (failed(matchInts(op.getEdgePaddingHigh(), edgePaddingHigh)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant edge_padding_high");
    if (failed(matchInts(op.getInteriorPadding(), interiorPadding)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant interior_padding");

    SmallVector<Type> inferredReturnTypes;
    if (failed(hlo::inferPadOp(
            /*location=*/{}, op.getOperand().getType(),
            op.getPaddingValue().getType(),
            rewriter.getI64TensorAttr(edgePaddingLow),
            rewriter.getI64TensorAttr(edgePaddingHigh),
            rewriter.getI64TensorAttr(interiorPadding), inferredReturnTypes)))
      return rewriter.notifyMatchFailure(op, "inferPadOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnTypes);
  }
};

struct RefineDynamicReshapeOpPattern
    : public OpRewritePattern<DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getOutputShape());
  }
};

struct RefineInferTypeOpInterfacePattern
    : public OpInterfaceRewritePattern<InferTypeOpInterface> {
  explicit RefineInferTypeOpInterfacePattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(InferTypeOpInterface op,
                                PatternRewriter& rewriter) const override {
    // Unlike in TensorFlow's type inference pass, here we work only with
    // allowlisted ops to focus our support on well-defined semantics of
    // StableHLO programs.
    if (!isa<chlo::ChloDialect, StablehloDialect>(op->getDialect()))
      return rewriter.notifyMatchFailure(op, "unsupported dialect");

    // For the ops that implement InferTypeOpInterface, we reinfer their return
    // types and see what happens.
    // Operands of these ops might have been refined elsewhere (e.g. someone
    // might have updated argument types of a function) or earlier during this
    // pass, and this might enable refinement opportunities downstream.
    SmallVector<Type> inferredReturnTypes;
    if (failed(op.inferReturnTypes(getContext(), /*location=*/{},
                                   op->getOperands(), op->getAttrDictionary(),
                                   op->getRegions(), inferredReturnTypes)))
      return rewriter.notifyMatchFailure(op, "inferReturnTypes failed");
    return refineReturnTypes(rewriter, op, inferredReturnTypes);
  }
};

struct RefineRealDynamicSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Alternative #1: All attributes are fully static (SliceOp style).
    SmallVector<int64_t> startIndices, limitIndices, strides;
    if (succeeded(matchInts(op.getStartIndices(), startIndices)) &&
        succeeded(matchInts(op.getLimitIndices(), limitIndices)) &&
        succeeded(matchInts(op.getStrides(), strides))) {
      SmallVector<Type> inferredReturnTypes;
      if (failed(hlo::inferSliceOp(/*location=*/{}, op.getOperand().getType(),
                                   rewriter.getI64TensorAttr(startIndices),
                                   rewriter.getI64TensorAttr(limitIndices),
                                   rewriter.getI64TensorAttr(strides),
                                   inferredReturnTypes)))
        return rewriter.notifyMatchFailure(op, "inferSliceOp failed");
      return refineReturnTypes(rewriter, op, inferredReturnTypes);
    }

    // Alternative #2: Slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    if (matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) ||
        matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices))) {
      SmallVector<int64_t> strides;
      if (!succeeded(matchInts(op.getStrides(), strides)) ||
          !llvm::all_of(strides, [&](int64_t stride) { return stride == 1; }))
        return rewriter.notifyMatchFailure(op, "expected unit strides");

      // RealDynamicSliceOp::start_indices is a 1-dimensional tensor.
      // DynamicSliceOp::start_indices is a vararg of 0-dimensional tensors.
      // Adapt accordingly in order to be compatible with inferDynamicSliceOp.
      auto startIndicesElementType =
          op.getStartIndices().getType().cast<ShapedType>().getElementType();
      SmallVector<Type> startIndicesTypes(
          sliceSizesAttr.size(),
          RankedTensorType::get({}, startIndicesElementType));

      // RealDynamicSliceOp can take tensors of integer or index element types.
      // DynamicSliceOp::slice_sizes only supports i64 element type.
      // Adapt accordingly in order to be compatible with inferDynamicSliceOp.
      SmallVector<int64_t> sliceSizes;
      for (auto element : sliceSizesAttr.getValues<APInt>()) {
        sliceSizes.push_back(element.getSExtValue());
      }

      SmallVector<ShapedTypeComponents> inferredReturnTypes;
      if (failed(hlo::inferDynamicSliceOp(
              op.getLoc(), op.getOperand().getType(), startIndicesTypes,
              rewriter.getI64TensorAttr(sliceSizes), inferredReturnTypes)))
        return rewriter.notifyMatchFailure(op, "inferDynamicSliceOp failed");
      return refineReturnTypes(rewriter, op, inferredReturnTypes);
    }

    return rewriter.notifyMatchFailure(
        op,
        "expected either fully static attributes (SliceOp style) "
        "or static sliceSizes (DynamicSliceOp style)");
  }
};

struct RefineReduceScatterOpPattern : public OpRewritePattern<ReduceScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceScatterOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType().cast<ShapedType>();
    if (!operandType.hasRank())
      return rewriter.notifyMatchFailure(op, "expected ranked operand type");

    // This represents the cross_replica_and_partition process grouping strategy
    // that requires num_partitions to compute shardCount. Since we don't know
    // num_partitions at this point, we error out.
    if (op.getChannelHandle() && !op.getUseGlobalDeviceIds())
      return rewriter.notifyMatchFailure(op, "unsupported strategy");
    DenseIntElementsAttr replicaGroups = op.getReplicaGroups();
    auto shardCount = replicaGroups.getType().getDimSize(1);

    SmallVector<int64_t> refinement(operandType.getShape());
    if (!operandType.isDynamicDim(op.getScatterDimension()))
      refinement[op.getScatterDimension()] /= shardCount;
    return refineReturnShape(rewriter, op, refinement);
  }
};

struct RefineRngOpPattern : public OpRewritePattern<RngOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RngOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getShape());
  }
};

struct RefineUniformQuantizeOpPattern
    : public OpRewritePattern<UniformQuantizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UniformQuantizeOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferUniformQuantizeOp(
            /*location=*/{}, op.getOperand(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvertOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineWhileOpPattern : public OpRewritePattern<WhileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    // Push the potentially refined operand types into the nested regions.
    // This can lead to refinements of the return types of the body (but not
    // of the cond since it always returns tensor<i1>), but the key insight here
    // is that the enclosing while op doesn't care about these refinements
    // (because its return types are equal to its operand types).
    // If we end up with incompatibilities between while's return types and
    // body's return types, the verifier will tell us about that. This means
    // that the original program wasn't well-formed. TODO(burmako): Implement
    // better error reporting for this case.
    // This serves the current use cases well, so the implementation of more
    // sophisticated refinement algorithm is left for future work.
    rewriter.startRootUpdate(op);
    auto condStatus = refineValues(rewriter, op, op.getCond().getArguments(),
                                   op.getOperandTypes());
    auto bodyStatus = refineValues(rewriter, op, op.getBody().getArguments(),
                                   op.getOperandTypes());
    if (succeeded(condStatus) || succeeded(bodyStatus)) {
      rewriter.finalizeRootUpdate(op);
      return success();
    } else {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
  }
};

struct UpdateFunctionTypePattern : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter& rewriter) const override {
    // Check whether any of the values returned by `func.return` are
    // `tensor.cast` ops which cast more specific type to less specific type.
    // Such ops are produced by the algorithm behind this pass to avoid
    // bringing the enclosing `func.func` op into an inconsistent state when
    // refining individual ops. This pattern cleans this up.
    bool needsUpdate = false;
    SmallVector<Type> updatedResultTypes(op.getOperandTypes());
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      auto cast = dyn_cast_or_null<tensor::CastOp>(operand.getDefiningOp());
      if (!cast) continue;

      // Only proceed if the type that we're casting from is more specific
      // than the type that we're casting to.
      auto sourceType = cast.getSource().getType();
      auto destType = cast.getType();
      auto mostSpecificType = hlo::inferMostSpecificType(
          /*location=*/{}, {sourceType, destType});
      if (failed(mostSpecificType) || destType == *mostSpecificType) continue;

      // If the source type of the cast is more specific than the target type,
      // then we conclude that the cast is redundant (i.e. needs to be removed)
      // and that the return type of the function needs an update.
      needsUpdate = true;
      updatedResultTypes[i] = sourceType;
      rewriter.replaceOp(cast, cast->getOperands());
    }
    if (!needsUpdate)
      return rewriter.notifyMatchFailure(op, "doesn't need update");

    // If the type of the enclosing `func.func` needs an update, we simply
    // call setType. We can afford this simplicity because our algorithm
    // currently supports only one function per module.
    auto func = cast<func::FuncOp>(op->getParentOp());
    func.setType(
        rewriter.getFunctionType(func.getArgumentTypes(), updatedResultTypes));
    return success();
  }
};

struct UpdateRegionTypePattern : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter& rewriter) const override {
    if (!isa<CaseOp, IfOp>(op->getParentOp()))
      return rewriter.notifyMatchFailure(op, "unsupported region");

    bool needsUpdate = false;
    SmallVector<Type> updatedResultTypes(op.getOperandTypes());
    for (auto [regionType, refinedType] : llvm::zip(
             op->getParentOp()->getResultTypes(), op->getOperandTypes())) {
      auto mostSpecificType = hlo::inferMostSpecificType(
          /*location=*/{}, {regionType, refinedType});
      if (failed(mostSpecificType) || regionType == *mostSpecificType) continue;
      needsUpdate = true;
    }
    if (!needsUpdate)
      return rewriter.notifyMatchFailure(op, "doesn't need update");

    rewriter.updateRootInPlace(op->getParentOp(), [&]() { return; });
    return success();
  }
};

struct StablehloRefineShapesPass
    : public impl::StablehloRefineShapesPassBase<StablehloRefineShapesPass> {
  using StablehloRefineShapesPassBase::StablehloRefineShapesPassBase;

  void runOnOperation() override {
    // Only one function per module is supported at the moment to avoid the need
    // to think about iterative type inference algorithms.
    // Current use cases are served well by inlining multiple functions into
    // a single function, so we leave native support for multiple functions to
    // future work.
    ModuleOp module = getOperation();
    auto funcs = llvm::to_vector(module.getOps<func::FuncOp>());
    if (funcs.empty()) return;
    if (funcs.size() != 1) {
      module.emitOpError() << "must have no more than one function";
      return signalPassFailure();
    }

    // Similarly, only one block per function is supported at the moment.
    // At the StableHLO level, functions are expected to only have one block,
    // so supporting more is out of scope for this pass.
    func::FuncOp func = funcs[0];
    if (!func.getRegion().hasOneBlock()) {
      func.emitOpError() << "must have exactly one block";
      return signalPassFailure();
    }

    // The algorithm behind this pass consists of a single traversal of the
    // function. This is sufficient because we only support one function per
    // program at the moment.
    // TODO(#1048): Find out why .maxIterations = 1 no longer works.
    // There have been recent refactorings in applyPatternsAndFoldGreedily
    // upstream, and that might be the reason.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 2;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::AnyOp;

    RewritePatternSet patterns(&getContext());
    patterns.add<EvalAddOpPattern>(&getContext());
    patterns.add<EvalAndOpPattern>(&getContext());
    patterns.add<EvalBroadcastInDimOpPattern>(&getContext());
    patterns.add<EvalCompareOpPattern>(&getContext());
    patterns.add<EvalConcatenateOpPattern>(&getContext());
    patterns.add<EvalConvertOpPattern>(&getContext());
    patterns.add<EvalDivOpPattern>(&getContext());
    patterns.add<EvalGetDimensionSizeOpPattern>(&getContext());
    patterns.add<EvalMaxOpPattern>(&getContext());
    patterns.add<EvalMulOpPattern>(&getContext());
    patterns.add<EvalRemOpPattern>(&getContext());
    patterns.add<EvalReshapeOpPattern>(&getContext());
    patterns.add<EvalSelectOpPattern>(&getContext());
    patterns.add<EvalSignOpPattern>(&getContext());
    patterns.add<EvalSliceOpPattern>(&getContext());
    patterns.add<EvalSubtractOpPattern>(&getContext());
    patterns.add<RefineAllGatherOpPattern>(&getContext());
    patterns.add<RefineBitcastConvertOpPattern>(&getContext());
    patterns.add<RefineConvertOpPattern>(&getContext());
    patterns.add<RefineConvolutionOpPattern>(&getContext());
    patterns.add<RefineCustomCallOpPattern>(&getContext());
    patterns.add<RefineDotGeneralOpPattern>(&getContext());
    patterns.add<RefineDynamicBroadcastInDimOpPattern>(&getContext());
    patterns.add<RefineDynamicConvOpPattern>(&getContext());
    patterns.add<RefineDynamicIotaOpPattern>(&getContext());
    patterns.add<RefineDynamicPadOpPattern>(&getContext());
    patterns.add<RefineDynamicReshapeOpPattern>(&getContext());
    patterns.add<RefineInferTypeOpInterfacePattern>(&getContext());
    patterns.add<RefineRealDynamicSliceOpPattern>(&getContext());
    patterns.add<RefineReduceScatterOpPattern>(&getContext());
    patterns.add<RefineRngOpPattern>(&getContext());
    patterns.add<RefineUniformQuantizeOpPattern>(&getContext());
    patterns.add<RefineWhileOpPattern>(&getContext());
    patterns.add<UpdateFunctionTypePattern>(&getContext());
    patterns.add<UpdateRegionTypePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace stablehlo
}  // namespace mlir
