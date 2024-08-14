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
#include "mlir/Dialect/Quant/QuantOps.h"
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
Value buildRescale(PatternRewriter& rewriter, Location loc,
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
Value buildRescaleToInt32(PatternRewriter& rewriter, Location loc,
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
Value buildRescaleFromInt32(PatternRewriter& rewriter, Location loc,
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

template <typename StablehloOp>
LogicalResult matchAndRewriteAddSub(StablehloOp op, PatternRewriter& rewriter) {
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

  // Following quantization described in tflite
  // In details it does:
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

  double lhsRescaleScale = lhsScale / maxScale2x;
  double rhsRescaleScale = rhsScale / maxScale2x;
  double resultRescaleScale =
      maxScale2x / (resultScale * static_cast<double>(1 << inputShift));

  auto loc = op.getLoc();

  // Implement single rounding only
  Value rescaledLhs = buildRescaleToInt32(
      rewriter, loc, lhs,
      /*scale=*/lhsRescaleScale * static_cast<double>(1 << inputShift),
      lhsQType.getZeroPoint());
  Value rescaledRhs = buildRescaleToInt32(
      rewriter, loc, rhs,
      /*scale=*/rhsRescaleScale * static_cast<double>(1 << inputShift),
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

template <typename StablehloOpType>
struct QuantizedStablehloOpConversion
    : public OpRewritePattern<StablehloOpType> {
  using OpRewritePattern<StablehloOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(StablehloOpType op,
                                PatternRewriter& rewriter) const override {
    return matchAndRewriteAddSub<StablehloOpType>(op, rewriter);
  }
};

struct StablehloQuantLegalizeToTosaRescalePass
    : impl::StablehloQuantLegalizeToTosaRescalePassBase<
          StablehloQuantLegalizeToTosaRescalePass> {
  LogicalResult initialize(MLIRContext* ctx) override {
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
    RewritePatternSet* patterns, MLIRContext* context) {
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::AddOp>>(
      {"StablehloQuantAddOp"}, context);
  patterns->addWithLabel<QuantizedStablehloOpConversion<stablehlo::SubtractOp>>(
      {"StablehloQuantSubtractOp"}, context);
}

}  // namespace tosa
}  // namespace mlir
