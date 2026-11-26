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

// Implements logic for lowering StableHLO pointwise ops to Linalg dialect.
// These patterns are separated out to their own file to save on the compilation
// times, given that we instantiate a large number of class templates here.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "stablehlo-conversions"

namespace mlir::stablehlo {
namespace {
int64_t getRank(Value v) { return cast<ShapedType>(v.getType()).getRank(); }

int64_t getMaxRank(ValueRange operands) {
  int64_t maxRank = 0;
  for (Value operand : operands) {
    maxRank = std::max(maxRank, getRank(operand));
  }
  return maxRank;
}

bool isScalar(Value v) { return getRank(v) == 0; }

/// Inserts block arguments in places where scalar inputs have a nullptr.
SmallVector<Value> interleaveScalarAndBlockArgs(ValueRange scalarInputs,
                                                ValueRange blockArgs) {
  SmallVector<Value> result;
  auto argsIter = blockArgs.begin();
  for (Value scalarInput : scalarInputs) {
    if (scalarInput) {
      result.push_back(scalarInput);
    } else {
      result.push_back(*argsIter);
      ++argsIter;
    }
  }
  return result;
}

struct PointwiseConversionInfo {
  int64_t maxOperandRank = 0;
  ShapedType resultType;
};

/// Checks the preconditions for conversion of pointwise HLO ops to linalg.
/// Returns the max operand rank and the result type on success.
FailureOr<PointwiseConversionInfo> checkOperandsAndResults(
    Operation* op, ValueRange operands, const TypeConverter& typeConverter,
    ConversionPatternRewriter& rewriter) {
  int64_t maxRank = getMaxRank(operands);

  // Apply only if all operands are scalar or have the same rank. Some ops,
  // like `stablehlo.select`, support implicit broadcasting of scalars.
  if (!llvm::all_of(operands, [&](Value v) {
        int64_t r = getRank(v);
        return r == 0 || r == maxRank;
      })) {
    return rewriter.notifyMatchFailure(
        op, "Operands must be of same rank or scalar.");
  }

  // Find result type, if on tensors.
  auto resultTy = cast_or_null<RankedTensorType>(
      typeConverter.convertType(op->getResultTypes().front()));

  // Check result type compatibility.
  if (!resultTy || resultTy.getRank() != maxRank ||
      !(resultTy.getElementType().isSignlessIntOrFloat() ||
        isa<ComplexType>(resultTy.getElementType()))) {
    return rewriter.notifyMatchFailure(
        op, "mismatched operand/result types or iterator count");
  }

  // All-scalar pointwise ops inside of linalg ops are processes by
  // ScalarHloToArithmeticPattern.
  if (maxRank == 0 && isInBodyOfLinalgOps(op)) return failure();

  return PointwiseConversionInfo{maxRank, resultTy};
}

/// If `input` is a splat constant value, materialize the scalar splat
/// value. Otherwise, return nullopt.
std::optional<Value> materializeSplatScalarConstant(RewriterBase& rewriter,
                                                    Location loc, Value input) {
  SplatElementsAttr attr;
  Type elementType = mlir::getElementTypeOrSelf(input.getType());
  if (!matchPattern(input, m_Constant(&attr))) return {};
  if (isa<IntegerType, FloatType, IndexType>(elementType)) {
    return arith::ConstantOp::create(rewriter, loc, elementType,
                                     attr.getSplatValue<TypedAttr>())
        .getResult();
  }
  if (isa<ComplexType>(elementType)) {
    return complex::ConstantOp::create(rewriter, loc, elementType,
                                       attr.getSplatValue<ArrayAttr>())
        .getResult();
  }
  return {};
}

/// Returns discardable attributes on `op` excluding `elidedAttrs`.
SmallVector<NamedAttribute> getPointwisePrunedAttrs(
    Operation* op, ArrayRef<StringRef> elidedAttrs) {
  SmallVector<NamedAttribute> attrs;
  for (NamedAttribute attr : op->getDiscardableAttrDictionary().getValue()) {
    if (!llvm::is_contained(elidedAttrs, attr.getName().strref()))
      attrs.push_back(attr);
  }
  return attrs;
}

/// Converts a HLO operation to a linalg.map or linalg.generic op that contains
/// the corresponding scalar operations.
struct PointwiseToLinalgPattern final : ConversionPattern {
  PointwiseToLinalgPattern(TypeConverter& converter, StringRef rootName,
                           ArrayRef<StringRef> elidedAttrs,
                           MLIRContext* context, bool enablePrimitiveOps,
                           bool captureScalarInputs)
      : ConversionPattern(converter, rootName, /*benefit=*/1, context),
        elidedAttrs(elidedAttrs),
        enablePrimitiveOps(enablePrimitiveOps),
        captureScalarInputs(captureScalarInputs) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    auto conversionInfo =
        checkOperandsAndResults(op, operands, *getTypeConverter(), rewriter);
    if (failed(conversionInfo)) {
      return failure();
    }

    int64_t maxRank = conversionInfo->maxOperandRank;
    ShapedType resultTy = conversionInfo->resultType;
    Location loc = op->getLoc();

    // Find input/output values and types.
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, operands);

    // Mapped inputs are cast to the same shape as the init tensor.
    // Values from scalar inputs are extracted and used directly in the block.
    SmallVector<Value> mappedInputs;
    SmallVector<Value> scalarInputs;
    for (Value input : operands) {
      if (std::optional<Value> splatVal =
              materializeSplatScalarConstant(rewriter, loc, input)) {
        scalarInputs.push_back(*splatVal);
      } else if (getRank(input) == maxRank) {
        mappedInputs.push_back(coerceTensorShape(
            rewriter, loc, cast<TypedValue<ShapedType>>(input),
            cast<ShapedType>(emptyTensor.getType())));
        scalarInputs.push_back(nullptr);
      } else if (captureScalarInputs) {
        scalarInputs.push_back(tensor::ExtractOp::create(rewriter, loc, input));
      } else {
        mappedInputs.push_back(input);
        scalarInputs.push_back(nullptr);
      }
    }

    if (enablePrimitiveOps) {
      bool failed = false;
      Operation* mapOp = linalg::MapOp::create(
          rewriter, loc, mappedInputs, emptyTensor,
          [&](OpBuilder& b, Location blockLoc, ValueRange args) {
            Value innerResult = StablehloOpToStdScalarOp::mapOp(
                op, getElementTypeOrSelf(emptyTensor),
                interleaveScalarAndBlockArgs(scalarInputs, args), &b);
            if (!innerResult) {
              failed = true;
              return;
            }
            linalg::YieldOp::create(b, blockLoc, innerResult);
          },
          getPointwisePrunedAttrs(op, elidedAttrs));
      if (failed) return failure();
      rewriter.replaceOp(op, mapOp->getResults());
      return success();
    }

    // Create indexing maps.
    AffineMap scalarMap = AffineMap::get(maxRank, 0, rewriter.getContext());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(maxRank);
    SmallVector<AffineMap> maps;
    for (Value v : mappedInputs)
      maps.push_back(isScalar(v) ? scalarMap : idMap);
    maps.push_back(idMap);
    bool failed = false;
    Operation* linalgOp = linalg::GenericOp::create(
        rewriter, loc, emptyTensor.getType(), mappedInputs, emptyTensor, maps,
        getNParallelLoopsAttrs(maxRank),
        [&](OpBuilder& nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          Type innerResultTy = getElementTypeOrSelf(emptyTensor);
          auto argvec =
              llvm::to_vector<2>(args.take_front(mappedInputs.size()));
          Value semiring = preSparsify(op, argvec, innerResultTy, &rewriter);
          Value innerResult = StablehloOpToStdScalarOp::mapOp(
              op, innerResultTy,
              interleaveScalarAndBlockArgs(scalarInputs, args), &rewriter);
          if (!innerResult) {
            failed = true;
          } else {
            innerResult = postSparsify(op, semiring, innerResult, &rewriter);
            linalg::YieldOp::create(nestedBuilder, loc, innerResult);
          }
        },
        getPointwisePrunedAttrs(op, elidedAttrs));
    if (failed) return failure();

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }

  ArrayRef<StringRef> elidedAttrs;
  bool enablePrimitiveOps;
  bool captureScalarInputs;
};

void addPattern(RewritePatternSet* patterns,
                std::unique_ptr<RewritePattern> pattern) {
  patterns->add(std::move(pattern));
}

template <typename... OpTypes>
void populatePointwisePatterns(MLIRContext* context,
                               TypeConverter& typeConverter,
                               RewritePatternSet* patterns,
                               bool enablePrimitiveOps,
                               bool captureScalarInputs) {
  (addPattern(patterns, std::make_unique<PointwiseToLinalgPattern>(
                            typeConverter, OpTypes::getOperationName(),
                            OpTypes::getAttributeNames(), context,
                            enablePrimitiveOps, captureScalarInputs)),
   ...);
}
}  // namespace

namespace detail {
void populatePointwiseStablehloToLinalgConversionPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns, bool enablePrimitiveOps,
    bool captureScalarInputs) {
  populatePointwisePatterns<
      AbsOp, AddOp, AndOp, Atan2Op, BitcastConvertOp, CbrtOp, CeilOp, ClampOp,
      ClzOp, CompareOp, ComplexOp, ConvertOp, CosineOp, DivOp, ExpOp, Expm1Op,
      Exp2Op, FloorOp, FmaOp, ImagOp, IsFiniteOp, Log1pOp, Log2Op, LogOp,
      LogisticOp, MaxOp, MinOp, MulOp, NegOp, NotOp, OrOp, PopulationCountOp,
      PowOp, RealOp, ReducePrecisionOp, RemOp, RoundNearestEvenOp, RoundOp,
      RsqrtOp, SelectOp, ShiftLeftOp, ShiftRightArithmeticOp,
      ShiftRightLogicalOp, SignOp, SineOp, SqrtOp, SubtractOp, TanOp, TanhOp,
      XorOp>(context, typeConverter, patterns, enablePrimitiveOps,
             captureScalarInputs);
}
}  // namespace detail
}  // namespace mlir::stablehlo
