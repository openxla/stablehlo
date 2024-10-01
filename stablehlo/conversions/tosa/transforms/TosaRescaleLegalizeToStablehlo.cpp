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

#include <limits>
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

#define PASS_NAME "tosa-rescale-legalize-to-stablehlo"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {

#define GEN_PASS_DEF_TOSARESCALELEGALIZETOSTABLEHLOPASS
#include "stablehlo/conversions/tosa/transforms/Passes.h.inc"

namespace {

Value getStablehloConstantOp(PatternRewriter& rewriter, Location loc,
                             const DenseElementsAttr& attr) {
  return rewriter.create<stablehlo::ConstantOp>(loc, attr.getType(), attr);
}

struct ConvertTosaRescaleToStablehlo : public OpRewritePattern<RescaleOp> {
  using OpRewritePattern<RescaleOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(RescaleOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult ConvertTosaRescaleToStablehlo::matchAndRewrite(
    RescaleOp op, PatternRewriter& rewriter) const {
  Value input = op.getInput();
  auto loc = op.getLoc();
  auto inputType = dyn_cast<ShapedType>(op.getInput().getType());
  auto outputType = dyn_cast<ShapedType>(op.getOutput().getType());

  if (!inputType || !outputType) {
    return rewriter.notifyMatchFailure(
        op, "input and output should have shaped tensor types");
  }

  bool scale32 = op.getScale32();
  bool doubleRound = op.getDoubleRound();
  bool perChannel = op.getPerChannel();

  if (perChannel || doubleRound || !scale32) {
    return rewriter.notifyMatchFailure(
        op,
        "per_channel, double_round, or scale32=false are not yet supported");
  }

  auto inputEType = inputType.getElementType();
  auto outputEType = outputType.getElementType();
  auto inputQType = dyn_cast<quant::UniformQuantizedType>(inputEType);
  auto outputQType = dyn_cast<quant::UniformQuantizedType>(outputEType);

  if (inputQType) {
    // first bit_cast input to quantized storage type
    auto bitCastType = inputType.clone(inputQType.getStorageType());
    input =
        rewriter.create<stablehlo::BitcastConvertOp>(loc, bitCastType, input);
    // change inputType and inputEType to be based on inputQType's storage type
    inputEType = inputQType.getStorageType();
    inputType = inputType.clone(inputEType);
  }
  if (outputQType) {
    // change outputType and outputEType to be based on outputQType's storage
    // type
    outputEType = outputQType.getStorageType();
    outputType = outputType.clone(outputEType);
  }

  if (!inputEType.isInteger() || !outputEType.isInteger()) {
    return rewriter.notifyMatchFailure(
        op,
        "input and output element types must be integer types or quantized "
        "integer types");
  }

  auto i8Type = inputType.clone(rewriter.getI8Type());
  auto i32Type = inputType.clone(rewriter.getI32Type());
  auto i64Type = inputType.clone(rewriter.getI64Type());

  // construct multiplier, shift constant values from op attrs
  // for scale32, multiplier is tensor of i32
  Value multiplier = getStablehloConstantOp(
      rewriter, loc, DenseElementsAttr::get(i32Type, op.getMultiplier()));
  Value shift = getStablehloConstantOp(
      rewriter, loc, DenseElementsAttr::get(i8Type, op.getShift()));

  // construct inputZp and outputZp from op attrs
  Value inputZpI32 = getStablehloConstantOp(
      rewriter, loc, DenseElementsAttr::get(i32Type, op.getInputZpAttr()));
  Value outputZpI32 = getStablehloConstantOp(
      rewriter, loc, DenseElementsAttr::get(i32Type, op.getOutputZpAttr()));

  // construct constant 1, min and max tensors
  Value onesI64 = getStablehloConstantOp(rewriter, loc,
                                         DenseElementsAttr::get(i64Type, {1L}));

  // find min and max clamp values based on bitwidth of output element type
  unsigned outputBitWidth = outputEType.getIntOrFloatBitWidth();
  int64_t minOutputValue =
      APInt::getSignedMinValue(outputBitWidth).getSExtValue();
  int64_t maxOutputValue =
      APInt::getSignedMaxValue(outputBitWidth).getSExtValue();
  if (outputEType.isUnsignedInteger()) {
    minOutputValue = APInt::getMinValue(outputBitWidth).getZExtValue();
    maxOutputValue = APInt::getMaxValue(outputBitWidth).getZExtValue();
  }

  Value outputMinI64 = getStablehloConstantOp(
      rewriter, loc, DenseElementsAttr::get(i64Type, {minOutputValue}));
  Value outputMaxI64 = getStablehloConstantOp(
      rewriter, loc, DenseElementsAttr::get(i64Type, {maxOutputValue}));

  // convert to i64 tensors
  Value multiplierI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, multiplier);
  Value shiftI64 = rewriter.create<stablehlo::ConvertOp>(loc, i64Type, shift);
  Value inputZpI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, inputZpI32);
  Value outputZpI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, outputZpI32);

  Value inputI64 = rewriter.create<stablehlo::ConvertOp>(loc, i64Type, input);

  Value adjustedInput =
      rewriter.create<stablehlo::SubtractOp>(loc, inputI64, inputZpI64);
  Value adjustedShift =
      rewriter.create<stablehlo::SubtractOp>(loc, shiftI64, onesI64);

  Value round =
      rewriter.create<stablehlo::ShiftLeftOp>(loc, onesI64, adjustedShift);

  Value r1 =
      rewriter.create<stablehlo::MulOp>(loc, adjustedInput, multiplierI64);
  Value r2 = rewriter.create<stablehlo::AddOp>(loc, r1, round);
  Value r3 =
      rewriter.create<stablehlo::ShiftRightArithmeticOp>(loc, r2, shiftI64);
  Value r4 = rewriter.create<stablehlo::AddOp>(loc, r3, outputZpI64);
  Value r5 =
      rewriter.create<stablehlo::ClampOp>(loc, outputMinI64, r4, outputMaxI64);

  Value result;
  if (outputQType) {
    // outputType has been converted to tensor of storage type by this point
    Value r6 = rewriter.create<stablehlo::ConvertOp>(loc, outputType, r5);
    auto originalOutputType = outputType.clone(outputQType);
    result = rewriter.create<stablehlo::BitcastConvertOp>(
        loc, originalOutputType, r6);
  } else {
    result = rewriter.create<stablehlo::ConvertOp>(loc, outputType, r5);
  }
  rewriter.replaceOp(op, {result});

  return success();
}

struct TosaRescaleLegalizeToStablehloPass
    : impl::TosaRescaleLegalizeToStablehloPassBase<
          TosaRescaleLegalizeToStablehloPass> {
  LogicalResult initialize(MLIRContext* ctx) override {
    RewritePatternSet patternList(ctx);
    patternList.addWithLabel<ConvertTosaRescaleToStablehlo>(
        {"ConvertTosaRescaleToStablehlo"}, ctx);
    patterns = std::move(patternList);
    return success();
  }
  void runOnOperation() final {
    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, patterns))) {
      func.emitError("Failed to apply TosaRescaleLegalizeToStablehlo pass ");
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
};

}  // namespace

}  // namespace tosa
}  // namespace mlir
