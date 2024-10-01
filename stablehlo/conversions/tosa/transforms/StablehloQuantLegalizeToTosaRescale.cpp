/* Copyright 2024 OpenXLA Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#define PASS_NAME "stablehlo-quant-legalize-to-tosa-rescale"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {

#define GEN_PASS_DEF_STABLEHLOQUANTLEGALIZETOTOSARESCALEPASS
#include "stablehlo/conversions/tosa/transforms/Passes.h.inc"

namespace {

// create a tosa rescale op and return its result value
Value buildRescale(PatternRewriter &rewriter, Location loc,
                   ShapedType outputType, Value inputVal, int32_t multiplier,
                   int32_t shift, int64_t inputZp, int64_t outputZp,
                   bool doubleRound, bool scale32, bool perChannel) {
  auto rescale_op = rewriter.create<RescaleOp>(
      loc, outputType, inputVal,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(inputZp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(outputZp)),
      rewriter.getDenseI32ArrayAttr({multiplier}),
      rewriter.getDenseI8ArrayAttr({static_cast<int8_t>(shift)}),
      rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(doubleRound),
      rewriter.getBoolAttr(perChannel));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Location loc,
                          Value inputVal, double inputScale, int64_t inputZp) {
  auto inputType = cast<ShapedType>(inputVal.getType());
  auto outputType = inputType.clone(rewriter.getI32Type());

  const int32_t scaleWidth = 32;
  int32_t multiplier, shift;
  computeMultiplierAndShift(inputScale, multiplier, shift, scaleWidth);

  return buildRescale(rewriter, loc, outputType, inputVal, multiplier, shift,
                      inputZp,
                      /*output_zp=*/0,
                      /*doubleRound=*/false,
                      /*scale32=*/true,
                      /*perChannel=*/false);
}

// Creates TOSA rescale op with int32 input
Value buildRescaleFromInt32(PatternRewriter &rewriter, Location loc,
                            ShapedType outputType, Value inputVal,
                            double outputScale, int64_t outputZp) {
  // Input should be int32 type
  auto inputType = cast<ShapedType>(inputVal.getType());
  (void)inputType;
  assert(inputType.getElementType().isInteger(32) &&
         "expected rescale input element type to be i32");

  const int32_t scaleWidth = 32;
  int32_t multiplier, shift;
  computeMultiplierAndShift(outputScale, multiplier, shift, scaleWidth);

  return buildRescale(rewriter, loc, outputType, inputVal, multiplier, shift,
                      /*input_zp=*/0, outputZp,
                      /*doubleRound=*/false,
                      /*scale32=*/true,
                      /*perChannel=*/false);
}

using UnaryRescaleScalesFn =
    void (*)(const quant::UniformQuantizedType &operandQType,
             const quant::UniformQuantizedType &resultQType,
             double &operandRescaleScale, double &resultRescaleScale);

void GetUnaryRescaleScales(const quant::UniformQuantizedType &operandQType,
                           const quant::UniformQuantizedType &resultQType,
                           double &operandRescaleScale,
                           double &resultRescaleScale) {
  double operandScale = operandQType.getScale();
  double resultScale = resultQType.getScale();

  // rescale inputs to I32 with scale=1.0
  // perform I32 unary operation
  // rescale result to scale = operandScale / resultScale

  operandRescaleScale = 1.0f;
  resultRescaleScale = operandScale / resultScale;
}

template <typename StablehloOp>
LogicalResult matchAndRewriteUnaryOp(
    StablehloOp op, PatternRewriter &rewriter,
    UnaryRescaleScalesFn rescaleScalesFn = GetUnaryRescaleScales) {
  Value operand = op.getOperand();
  Value result = op.getResult();

  auto operandType = cast<ShapedType>(operand.getType());
  auto resultType = cast<ShapedType>(result.getType());

  auto operandQType =
      dyn_cast<quant::UniformQuantizedType>(operandType.getElementType());
  auto resultQType =
      dyn_cast<quant::UniformQuantizedType>(resultType.getElementType());

  if (!operandQType || !resultQType) {
    return rewriter.notifyMatchFailure(
        op,
        "The conversion supports operands/results with per-tensor quantized "
        "types only");
  }

  double operandRescaleScale, resultRescaleScale;

  rescaleScalesFn(operandQType, resultQType, operandRescaleScale,
                  resultRescaleScale);

  auto loc = op.getLoc();

  // Implement single rounding only
  Value rescaledOperand = buildRescaleToInt32(
      rewriter, loc, operand, operandRescaleScale, operandQType.getZeroPoint());

  auto rescaledResultType = resultType.clone(rewriter.getI32Type());
  Value rescaledResult =
      rewriter.create<StablehloOp>(loc, rescaledResultType, rescaledOperand)
          .getResult();

  Value newOutput =
      buildRescaleFromInt32(rewriter, loc, resultType, rescaledResult,
                            resultRescaleScale, resultQType.getZeroPoint());

  rewriter.replaceOp(op, {newOutput});
  return success();
}

LogicalResult matchAndRewriteOp(stablehlo::AbsOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteUnaryOp(op, rewriter);
}

using BinaryRescaleScalesFn = void (*)(
    const quant::UniformQuantizedType &lhsQType,
    const quant::UniformQuantizedType &rhsQType,
    const quant::UniformQuantizedType &resultQType, double &lhsRescaleScale,
    double &rhsRescaleScale, double &resultRescaleScale);

void GetAddSubRescaleScales(const quant::UniformQuantizedType &lhsQType,
                            const quant::UniformQuantizedType &rhsQType,
                            const quant::UniformQuantizedType &resultQType,
                            double &lhsRescaleScale, double &rhsRescaleScale,
                            double &resultRescaleScale) {
  // 1. Rescale inputs to scale = 2.0 x max(lhs.scale, rhs.scale)
  // 2. Extra left shift to input to increase precision
  // Where input_shift = 20 if input is 8-bit
  // input_shift = 15 if input is 16-bit

  double lhsScale = lhsQType.getScale();
  double rhsScale = rhsQType.getScale();
  double resultScale = resultQType.getScale();
  double maxScale2x = 2.0 * std::max(lhsScale, rhsScale);

  const int32_t SHIFT_8_BIT = 20;
  const int32_t SHIFT_16_BIT = 15;

  int32_t inputShift = (resultQType.getStorageTypeIntegralWidth() == 16)
                           ? SHIFT_16_BIT
                           : SHIFT_8_BIT;

  lhsRescaleScale =
      (lhsScale / maxScale2x) * static_cast<double>(1 << inputShift);
  rhsRescaleScale =
      (rhsScale / maxScale2x) * static_cast<double>(1 << inputShift);
  resultRescaleScale =
      maxScale2x / (resultScale * static_cast<double>(1 << inputShift));
}

void GetMulDivRescaleScales(const quant::UniformQuantizedType &lhsQType,
                            const quant::UniformQuantizedType &rhsQType,
                            const quant::UniformQuantizedType &resultQType,
                            double &lhsRescaleScale, double &rhsRescaleScale,
                            double &resultRescaleScale) {
  double lhsScale = lhsQType.getScale();
  double rhsScale = rhsQType.getScale();
  double resultScale = resultQType.getScale();

  // rescale inputs to I32 with scale=1.0
  // perform I32 multiply or divide
  // rescale result to scale=(lhsScale * rhsScale) / resultScale

  lhsRescaleScale = 1.0f;
  rhsRescaleScale = 1.0f;
  resultRescaleScale = lhsScale * rhsScale / resultScale;
}

void GetMinMaxRescaleScales(const quant::UniformQuantizedType &lhsQType,
                            const quant::UniformQuantizedType &rhsQType,
                            const quant::UniformQuantizedType &resultQType,
                            double &lhsRescaleScale, double &rhsRescaleScale,
                            double &resultRescaleScale) {
  // 1. Rescale inputs to scale = max(lhs.scale, rhs.scale)
  // 2. Extra left shift to input to increase precision
  // Where input_shift = 20 if input is 8-bit
  // input_shift = 15 if input is 16-bit

  double lhsScale = lhsQType.getScale();
  double rhsScale = rhsQType.getScale();
  double resultScale = resultQType.getScale();

  double maxScale = std::max(lhsScale, rhsScale);

  const int32_t SHIFT_8_BIT = 20;
  const int32_t SHIFT_16_BIT = 15;

  int32_t inputShift = (resultQType.getStorageTypeIntegralWidth() == 16)
                           ? SHIFT_16_BIT
                           : SHIFT_8_BIT;

  lhsRescaleScale =
      (lhsScale / maxScale) * static_cast<double>(1 << inputShift);
  rhsRescaleScale =
      (rhsScale / maxScale) * static_cast<double>(1 << inputShift);
  resultRescaleScale =
      maxScale / (resultScale * static_cast<double>(1 << inputShift));
}

template <typename StablehloOp>
LogicalResult matchAndRewriteBinaryOp(StablehloOp op, PatternRewriter &rewriter,
                                      BinaryRescaleScalesFn rescaleScalesFn) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value result = op.getResult();

  auto lhsType = cast<ShapedType>(lhs.getType());
  auto rhsType = cast<ShapedType>(rhs.getType());
  auto resultType = cast<ShapedType>(result.getType());

  auto lhsQType =
      dyn_cast<quant::UniformQuantizedType>(lhsType.getElementType());
  auto rhsQType =
      dyn_cast<quant::UniformQuantizedType>(rhsType.getElementType());
  auto resultQType =
      dyn_cast<quant::UniformQuantizedType>(resultType.getElementType());

  if (!lhsQType || !rhsQType || !resultQType) {
    return rewriter.notifyMatchFailure(
        op,
        "The conversion supports operands/results with per-tensor quantized "
        "types only");
  }

  double lhsRescaleScale, rhsRescaleScale, resultRescaleScale;

  rescaleScalesFn(lhsQType, rhsQType, resultQType, lhsRescaleScale,
                  rhsRescaleScale, resultRescaleScale);

  auto loc = op.getLoc();

  // Implement single rounding only
  Value rescaledLhs = buildRescaleToInt32(rewriter, loc, lhs, lhsRescaleScale,
                                          lhsQType.getZeroPoint());
  Value rescaledRhs = buildRescaleToInt32(rewriter, loc, rhs, rhsRescaleScale,
                                          rhsQType.getZeroPoint());

  auto rescaledResultType = resultType.clone(rewriter.getI32Type());
  Value rescaledResult = rewriter
                             .create<StablehloOp>(loc, rescaledResultType,
                                                  rescaledLhs, rescaledRhs)
                             .getResult();

  Value newOutput =
      buildRescaleFromInt32(rewriter, loc, resultType, rescaledResult,
                            resultRescaleScale, resultQType.getZeroPoint());

  rewriter.replaceOp(op, {newOutput});
  return success();
}

LogicalResult matchAndRewriteOp(stablehlo::AddOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteBinaryOp(op, rewriter, GetAddSubRescaleScales);
}

LogicalResult matchAndRewriteOp(stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteBinaryOp(op, rewriter, GetAddSubRescaleScales);
}

LogicalResult matchAndRewriteOp(stablehlo::MulOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteBinaryOp(op, rewriter, GetMulDivRescaleScales);
}

LogicalResult matchAndRewriteOp(stablehlo::DivOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteBinaryOp(op, rewriter, GetMulDivRescaleScales);
}

LogicalResult matchAndRewriteOp(stablehlo::MinOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteBinaryOp(op, rewriter, GetMinMaxRescaleScales);
}

LogicalResult matchAndRewriteOp(stablehlo::MaxOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteBinaryOp(op, rewriter, GetMinMaxRescaleScales);
}

LogicalResult matchAndRewriteCompareOp(stablehlo::CompareOp op,
                                       PatternRewriter &rewriter) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value result = op.getResult();

  auto lhsType = cast<ShapedType>(lhs.getType());
  auto rhsType = cast<ShapedType>(rhs.getType());
  auto resultType = cast<ShapedType>(result.getType());

  auto lhsQType =
      dyn_cast<quant::UniformQuantizedType>(lhsType.getElementType());
  auto rhsQType =
      dyn_cast<quant::UniformQuantizedType>(rhsType.getElementType());

  if (!lhsQType || !rhsQType) {
    return rewriter.notifyMatchFailure(
        op,
        "The conversion supports operands with per-tensor quantized "
        "types only");
  }

  double lhsScale = lhsQType.getScale();
  double rhsScale = rhsQType.getScale();
  double maxScale = std::max(lhsScale, rhsScale);

  const int32_t SHIFT_8_BIT = 20;
  const int32_t SHIFT_16_BIT = 15;

  // note: compare op require lhs/rhs have equal base storage width
  int32_t inputShift = (lhsQType.getStorageTypeIntegralWidth() == 16)
                           ? SHIFT_16_BIT
                           : SHIFT_8_BIT;

  double lhsRescaleScale =
      (lhsScale / maxScale) * static_cast<double>(1 << inputShift);
  double rhsRescaleScale =
      (rhsScale / maxScale) * static_cast<double>(1 << inputShift);

  auto loc = op.getLoc();

  // Implement single rounding only
  Value rescaledLhs = buildRescaleToInt32(rewriter, loc, lhs, lhsRescaleScale,
                                          lhsQType.getZeroPoint());
  Value rescaledRhs = buildRescaleToInt32(rewriter, loc, rhs, rhsRescaleScale,
                                          rhsQType.getZeroPoint());

  auto compareDirection = op.getComparisonDirection();
  auto compareTypeAttr = op.getCompareTypeAttr();

  Value newOutput = rewriter
                        .create<stablehlo::CompareOp>(
                            loc, resultType, rescaledLhs, rescaledRhs,
                            compareDirection, compareTypeAttr)
                        .getResult();

  rewriter.replaceOp(op, {newOutput});
  return success();
}

LogicalResult matchAndRewriteOp(stablehlo::CompareOp op,
                                PatternRewriter &rewriter) {
  return matchAndRewriteCompareOp(op, rewriter);
}

template <typename StablehloOpType>
struct QuantizedStablehloOpConversion
    : public OpRewritePattern<StablehloOpType> {
  using OpRewritePattern<StablehloOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(StablehloOpType op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteOp(op, rewriter);
  }
};

struct StablehloQuantLegalizeToTosaRescalePass
    : impl::StablehloQuantLegalizeToTosaRescalePassBase<
          StablehloQuantLegalizeToTosaRescalePass> {
  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList(ctx);
    populateStablehloQuantLegalizeToTosaRescalePatterns(&patternList, ctx);
    patterns = std::move(patternList);
    return success();
  }
  void runOnOperation() final {
    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, patterns))) {
      func.emitError(
          "Failed to apply StablehloQuantLegalizeToTosaRescale pass ");
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
};

}  // namespace

void populateStablehloQuantLegalizeToTosaRescalePatterns(
    RewritePatternSet *patterns, MLIRContext *context) {
  // unary ops
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::AbsOp>>(
      {"StablehloQuantAbsOp"}, context);
  // binary ops
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::AddOp>>(
      {"StablehloQuantAddOp"}, context);
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::SubtractOp>>(
      {"StablehloQuantSubtractOp"}, context);
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::MulOp>>(
      {"StablehloQuantMulOp"}, context);
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::DivOp>>(
      {"StablehloQuantDivOp"}, context);
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::MaxOp>>(
      {"StablehloQuantMaxOp"}, context);
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::MinOp>>(
      {"StablehloQuantMinOp"}, context);
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::CompareOp>>(
      {"StablehloQuantCompareOp"}, context);
}

}  // namespace tosa
}  // namespace mlir
