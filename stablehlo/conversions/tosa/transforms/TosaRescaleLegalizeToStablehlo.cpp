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

#define PASS_NAME "tosa-rescale-legalize-to-stablehlo"
#define DEBUG_TYPE PASS_NAME

using namespace llvm;
using namespace mlir;
using namespace mlir::tosa;

namespace mlir {
namespace tosa {

#define GEN_PASS_DEF_TOSARESCALELEGALIZETOSTABLEHLOPASS
#include "stablehlo/conversions/tosa/transforms/Passes.h.inc"

namespace {

struct ConvertTosaRescaleToStablehlo
    : public OpRewritePattern<tosa::RescaleOp> {
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::RescaleOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult ConvertTosaRescaleToStablehlo::matchAndRewrite(
    tosa::RescaleOp op, PatternRewriter& rewriter) const {
  Value input = op.getInput();
  auto loc = op.getLoc();
  auto inputType = dyn_cast<mlir::ShapedType>(op.getInput().getType());
  auto outputType = dyn_cast<mlir::ShapedType>(op.getOutput().getType());

  if (!inputType || !outputType) {
    return rewriter.notifyMatchFailure(
        op, "input and output should have shaped tensor types");
  }

  bool scale32 = op.getScale32();
  bool doubleRound = op.getDoubleRound();
  bool perChannel = op.getPerChannel();

  if (perChannel || doubleRound || !scale32) {
    // do not support these modes yet
    return rewriter.notifyMatchFailure(
        op,
        "per_channel, double_round, or scale32=false are not yet supported");
  }

  auto inputEType = inputType.getElementType();
  auto outputEType = outputType.getElementType();
  auto inputQType = dyn_cast<mlir::quant::UniformQuantizedType>(inputEType);
  auto outputQType = dyn_cast<mlir::quant::UniformQuantizedType>(outputEType);

  if (inputQType) {
    // first bit_cast input to quantized storage type
    auto bitCastType = inputType.clone(inputQType.getStorageType());
    input =
        rewriter.create<stablehlo::BitcastConvertOp>(loc, bitCastType, input);
  }

  auto i8Type = inputType.clone(rewriter.getI8Type());
  auto i32Type = inputType.clone(rewriter.getI32Type());
  auto i64Type = inputType.clone(rewriter.getI64Type());

  // construct multiplier, shift values from op attrs
  auto multiplierAttr = DenseElementsAttr::get(i32Type, op.getMultiplier());
  Value multiplier =
      rewriter.create<stablehlo::ConstantOp>(loc, i32Type, multiplierAttr);

  auto shiftAttr = DenseElementsAttr::get(i8Type, op.getShift());
  Value shift = rewriter.create<stablehlo::ConstantOp>(loc, i8Type, shiftAttr);

  // construct inputZp and outputZp from op attrs
  int32_t inputZpValue = op.getInputZpAttr().getInt();
  auto inputZpAttr = DenseElementsAttr::get(i32Type, {inputZpValue});
  Value inputZp =
      rewriter.create<stablehlo::ConstantOp>(loc, i32Type, inputZpAttr);

  int32_t outputZpValue = op.getOutputZpAttr().getInt();
  auto outputZpAttr = DenseElementsAttr::get(i32Type, {outputZpValue});
  Value outputZp =
      rewriter.create<stablehlo::ConstantOp>(loc, i32Type, outputZpAttr);

  // construct constants 1, min and max tensors
  auto i64OnesAttr = DenseElementsAttr::get(i64Type, {1L});
  Value onesI64 =
      rewriter.create<stablehlo::ConstantOp>(loc, i64Type, i64OnesAttr);

  // find min and max clamp values based on bitwidth of output element type
  unsigned outputBitWidth = outputQType
                                ? outputQType.getStorageTypeIntegralWidth()
                                : outputEType.getIntOrFloatBitWidth();
  int32_t minOutputValue =
      APInt::getSignedMinValue(outputBitWidth).getSExtValue();
  int32_t maxOutputValue =
      APInt::getSignedMaxValue(outputBitWidth).getSExtValue();
  auto i32MinAttr = DenseElementsAttr::get(i32Type, {minOutputValue});
  Value outputMin =
      rewriter.create<stablehlo::ConstantOp>(loc, i32Type, i32MinAttr);
  auto i32MaxAttr = DenseElementsAttr::get(i32Type, {maxOutputValue});
  Value outputMax =
      rewriter.create<stablehlo::ConstantOp>(loc, i32Type, i32MaxAttr);

  // convert to i64 tensors
  Value multiplierI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, multiplier);
  Value shiftI64 = rewriter.create<stablehlo::ConvertOp>(loc, i64Type, shift);
  Value inputZpI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, inputZp);
  Value outputZpI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, outputZp);
  Value inputI64 = rewriter.create<stablehlo::ConvertOp>(loc, i64Type, input);
  Value outputMinI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, outputMin);
  Value outputMaxI64 =
      rewriter.create<stablehlo::ConvertOp>(loc, i64Type, outputMax);

  Value adjustedInput = rewriter.create<stablehlo::SubtractOp>(
      loc, i64Type, inputI64, inputZpI64);
  Value adjustedShift =
      rewriter.create<stablehlo::SubtractOp>(loc, i64Type, shiftI64, onesI64);

  Value round = rewriter.create<stablehlo::ShiftLeftOp>(loc, i64Type, onesI64,
                                                        adjustedShift);

  Value r1 = rewriter.create<stablehlo::MulOp>(loc, i64Type, adjustedInput,
                                               multiplierI64);
  Value r2 = rewriter.create<stablehlo::AddOp>(loc, i64Type, r1, round);
  Value r3 = rewriter.create<stablehlo::ShiftRightArithmeticOp>(loc, i64Type,
                                                                r2, shiftI64);
  Value r4 = rewriter.create<stablehlo::AddOp>(loc, i64Type, r3, outputZpI64);
  Value r5 = rewriter.create<stablehlo::ClampOp>(loc, i64Type, outputMinI64, r4,
                                                 outputMaxI64);

  Value result;
  if (outputQType) {
    auto storageType = outputType.clone(outputQType.getStorageType());
    Value r6 = rewriter.create<stablehlo::ConvertOp>(loc, storageType, r5);
    result = rewriter.create<stablehlo::BitcastConvertOp>(loc, outputType, r6);
  } else {
    result = rewriter.create<stablehlo::ConvertOp>(loc, outputType, r5);
  }
  rewriter.replaceOp(op, {result});

  return success();
}

struct TosaRescaleLegalizeToStablehloPass
    : impl::TosaRescaleLegalizeToStablehloPassBase<
          TosaRescaleLegalizeToStablehloPass> {
  void runOnOperation() final {
    auto* ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.addWithLabel<ConvertTosaRescaleToStablehlo>(
        {"ConvertTosaRescaleToStablehlo"}, ctx);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

}  // namespace tosa
}  // namespace mlir
