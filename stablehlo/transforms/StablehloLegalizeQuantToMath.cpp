/* Copyright 2024 The StableHLO Authors. All Rights Reserved.

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
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::stablehlo {
namespace {

using QuantType = quant::QuantizedType;
FailureOr<QuantType> getQuantType(Type type) {
  if (auto quantType =
          dyn_cast<quant::QuantizedType>(getElementTypeOrSelf(type)))
    return quantType;
  return failure();
}

bool isPerTensorType(Type type) {
  return isa<quant::UniformQuantizedType>(getElementTypeOrSelf(type));
}

bool isPerAxisType(Type type) {
  return isa<quant::UniformQuantizedPerAxisType>(getElementTypeOrSelf(type));
}

quant::UniformQuantizedType getPerTensorType(Type type) {
  return cast<quant::UniformQuantizedType>(getElementTypeOrSelf(type));
}

quant::UniformQuantizedPerAxisType getPerAxisType(Type type) {
  return cast<quant::UniformQuantizedPerAxisType>(getElementTypeOrSelf(type));
}

// Extracts scale and zero point info from input quant type info.
void getQuantizationParams(OpBuilder &builder, Location loc,
                           QuantType quantType, Value &scales,
                           Value &zeroPoints, bool outputZeroPointInFp,
                           DenseI64ArrayAttr &broadcastDims) {
  // Get scales/zero points for per-tensor and per-axis quantization cases.
  if (auto quantPerTensorType =
          dyn_cast<quant::UniformQuantizedType>(quantType)) {
    scales = builder.create<stablehlo::ConstantOp>(
        loc, builder.getF32FloatAttr(quantPerTensorType.getScale()));
    if (outputZeroPointInFp) {
      zeroPoints = builder.create<stablehlo::ConstantOp>(
          loc, builder.getF32FloatAttr(
                   static_cast<float>(quantPerTensorType.getZeroPoint())));
    } else {
      zeroPoints = builder.create<stablehlo::ConstantOp>(
          loc, builder.getI32IntegerAttr(
                   static_cast<int32_t>(quantPerTensorType.getZeroPoint())));
    }
  } else {
    auto quantPerAxisType = getPerAxisType(quantType);
    SmallVector<float> scalesVec;
    for (auto scale : quantPerAxisType.getScales()) scalesVec.push_back(scale);
    scales = builder.create<stablehlo::ConstantOp>(
        loc,
        DenseFPElementsAttr::get(
            RankedTensorType::get(
                {static_cast<int64_t>(quantPerAxisType.getScales().size())},
                builder.getF32Type()),
            scalesVec));
    if (outputZeroPointInFp) {
      SmallVector<float> zeroPointsVec;
      for (auto zeroPoint : quantPerAxisType.getZeroPoints())
        zeroPointsVec.push_back(zeroPoint);
      zeroPoints = builder.create<stablehlo::ConstantOp>(
          loc, DenseFPElementsAttr::get(
                   RankedTensorType::get(
                       {static_cast<int64_t>(
                           quantPerAxisType.getZeroPoints().size())},
                       builder.getF32Type()),
                   zeroPointsVec));
    } else {
      SmallVector<int32_t> zeroPointsVec;
      for (auto zeroPoint : quantPerAxisType.getZeroPoints())
        zeroPointsVec.push_back(zeroPoint);
      zeroPoints = builder.create<stablehlo::ConstantOp>(
          loc, DenseIntElementsAttr::get(
                   RankedTensorType::get(
                       {static_cast<int64_t>(
                           quantPerAxisType.getZeroPoints().size())},
                       builder.getI32Type()),
                   zeroPointsVec));
    }
    broadcastDims = DenseI64ArrayAttr::get(
        builder.getContext(),
        {static_cast<int64_t>(quantPerAxisType.getQuantizedDimension())});
  }
}

// Extracts storage min/max from input quant type info.
void getQuantizationStorageInfo(OpBuilder &builder, Location loc,
                                QuantType quantType, Value &storageMin,
                                Value &storageMax) {
  storageMin = builder.create<stablehlo::ConstantOp>(
      loc, builder.getF32FloatAttr(
               static_cast<float>(quantType.getStorageTypeMin())));
  storageMax = builder.create<stablehlo::ConstantOp>(
      loc, builder.getF32FloatAttr(
               static_cast<float>(quantType.getStorageTypeMax())));
}

// Extracts storage type of a UQ type, preserving its shape.
Type getQuantStorageType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    return shaped.clone(getQuantStorageType(shaped.getElementType()));
  }

  auto quantizedType = getQuantType(type);
  if (succeeded(quantizedType)) {
    return quantizedType->getStorageType();
  }
  return type;
}

Value applyMergedScalesAndZps(OpBuilder &builder, Location loc,
                              QuantType inputQuantType,
                              QuantType outputQuantType,
                              Value inputFloatTensor) {
  // Use single merged scale and merged zp if both input and output are
  // per-tensor quantized. Otherwise use a vector.
  if (isPerTensorType(inputQuantType) && isPerTensorType(outputQuantType)) {
    quant::UniformQuantizedType inputPerTensorType =
        getPerTensorType(inputQuantType);
    quant::UniformQuantizedType outputPerTensorType =
        getPerTensorType(outputQuantType);
    double mergedScaleFp =
        inputPerTensorType.getScale() / outputPerTensorType.getScale();
    auto mergedScale = builder.create<stablehlo::ConstantOp>(
        loc, builder.getF32FloatAttr(static_cast<float>(mergedScaleFp)));
    inputFloatTensor =
        builder.create<chlo::BroadcastMulOp>(loc, inputFloatTensor, mergedScale,
                                             /*broadcast_dimensions=*/nullptr);
    // Add merged_zp only when it is non-zero.
    double mergedZpFp = outputPerTensorType.getZeroPoint() -
                        inputPerTensorType.getZeroPoint() * mergedScaleFp;
    if (mergedZpFp != 0) {
      Value mergedZp = builder.create<stablehlo::ConstantOp>(
          loc, builder.getF32FloatAttr(static_cast<float>(mergedZpFp)));
      inputFloatTensor = builder.create<chlo::BroadcastAddOp>(
          loc, inputFloatTensor, mergedZp, /*broadcast_dimensions=*/nullptr);
    }
  } else {
    int64_t channelSize =
        isPerAxisType(outputQuantType)
            ? getPerAxisType(outputQuantType).getScales().size()
            : getPerAxisType(inputQuantType).getScales().size();
    int64_t quantizedDimension =
        isPerAxisType(outputQuantType)
            ? getPerAxisType(outputQuantType).getQuantizedDimension()
            : getPerAxisType(inputQuantType).getQuantizedDimension();
    SmallVector<double> mergedScaleDouble, mergedZpDouble;
    mergedScaleDouble.resize(channelSize);
    mergedZpDouble.resize(channelSize);
    for (int i = 0; i < channelSize; ++i) {
      mergedScaleDouble[i] =
          (isPerAxisType(inputQuantType)
               ? getPerAxisType(inputQuantType).getScales()[i]
               : getPerTensorType(inputQuantType).getScale()) /
          (isPerAxisType(outputQuantType)
               ? getPerAxisType(outputQuantType).getScales()[i]
               : getPerTensorType(outputQuantType).getScale());
      mergedZpDouble[i] =
          (isPerAxisType(outputQuantType)
               ? getPerAxisType(outputQuantType).getZeroPoints()[i]
               : getPerTensorType(outputQuantType).getZeroPoint()) -
          (isPerAxisType(inputQuantType)
               ? getPerAxisType(inputQuantType).getZeroPoints()[i]
               : getPerTensorType(inputQuantType).getZeroPoint()) *
              mergedScaleDouble[i];
    }
    SmallVector<float> mergedScaleFloat(mergedScaleDouble.begin(),
                                        mergedScaleDouble.end()),
        mergedZpFloat(mergedZpDouble.begin(), mergedZpDouble.end());

    auto broadcastDims =
        DenseI64ArrayAttr::get(builder.getContext(), {quantizedDimension});
    Value mergedScale = builder.create<stablehlo::ConstantOp>(
        loc, DenseFPElementsAttr::get(
                 RankedTensorType::get({channelSize}, builder.getF32Type()),
                 mergedScaleFloat));
    inputFloatTensor = builder.create<chlo::BroadcastMulOp>(
        loc, inputFloatTensor, mergedScale, broadcastDims);
    if (llvm::any_of(mergedZpFloat, [](double zp) { return zp != 0; })) {
      Value mergedZp = builder.create<stablehlo::ConstantOp>(
          loc, DenseFPElementsAttr::get(
                   RankedTensorType::get({channelSize}, builder.getF32Type()),
                   mergedZpFloat));
      inputFloatTensor = builder.create<chlo::BroadcastAddOp>(
          loc, inputFloatTensor, mergedZp, broadcastDims);
    }
  }
  return inputFloatTensor;
}

// This helper function create ops to requantize `input` tensor and returns the
// output tensor. Clamping is done if output integer bit-width < i32. It assumes
// that if both input and output tensor are per-axis quantized, they have the
// same quantization axis.
//
// Requantization is essentially dequantize --> quantize.
//
// Dequantize: (input - zp) * scale
// Quantize: input / scale + zp
//
// Hence,
//   output = (input - input_zp) * input_scale / output_scale + output_zp
//
// This is simplified as:
//   output = input * merged_scale + merged_zp
// where:
//   merged_zp = output_zp - input_zp * merged_scale.
//   merged_scale = input_scale / output_scale.
Value requantize(OpState op, Value input, QuantType inputQuantType,
                 QuantType outputQuantType, TensorType outputTensorType,
                 ConversionPatternRewriter &rewriter) {
  // Skip requantization when input and result have the same type.
  if (inputQuantType == outputQuantType) {
    return rewriter.create<stablehlo::ConvertOp>(op->getLoc(), outputTensorType,
                                                 input);
  }

  auto floatTensorType = outputTensorType.clone(rewriter.getF32Type());
  Value outputFloat = rewriter.create<stablehlo::ConvertOp>(
      op->getLoc(), floatTensorType, input);

  outputFloat = applyMergedScalesAndZps(rewriter, op->getLoc(), inputQuantType,
                                        outputQuantType, outputFloat);

  // Clamp output if the output integer bit-width <32.
  if (cast<IntegerType>(outputTensorType.getElementType()).getWidth() < 32) {
    Value quantizationMin, quantizationMax;
    getQuantizationStorageInfo(rewriter, op->getLoc(), outputQuantType,
                               quantizationMin, quantizationMax);
    // Clamp results by [quantizationMin, quantizationMax].
    outputFloat = rewriter.create<stablehlo::ClampOp>(
        op->getLoc(), quantizationMin, outputFloat, quantizationMax);
  }

  outputFloat = rewriter.create<stablehlo::RoundNearestEvenOp>(
      op->getLoc(), floatTensorType, outputFloat);
  return rewriter.create<stablehlo::ConvertOp>(op->getLoc(), outputTensorType,
                                               outputFloat);
}

class ConvertUniformQuantizeOp
    : public OpConversionPattern<stablehlo::UniformQuantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::UniformQuantizeOp op,
      stablehlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputElementType = getElementTypeOrSelf(op.getOperand().getType());
    if (inputElementType.isF32()) {
      auto quantType = getQuantType(op.getResult().getType());
      if (succeeded(quantType)) {
        return matchAndRewriteQuantize(op, adaptor, rewriter, *quantType);
      }
    } else if (isa<quant::UniformQuantizedType,
                   quant::UniformQuantizedPerAxisType>(inputElementType)) {
      auto inputQuantType = getQuantType(inputElementType);
      auto outputQuantType = getQuantType(op.getResult().getType());
      if (succeeded(inputQuantType) && succeeded(outputQuantType)) {
        if (isPerAxisType(*inputQuantType) && isPerAxisType(*outputQuantType) &&
            getPerAxisType(*inputQuantType).getQuantizedDimension() !=
                getPerAxisType(*outputQuantType).getQuantizedDimension()) {
          op->emitError("Cannot requantize while changing quantization_axis");
          return failure();
        }
        return matchAndRewriteRequantize(op, adaptor, rewriter, *inputQuantType,
                                         *outputQuantType);
      }
    }
    op->emitError("Unsupported input element type.");
    return failure();
  }

  LogicalResult matchAndRewriteQuantize(
      stablehlo::UniformQuantizeOp op,
      stablehlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, QuantType quantType) const {
    Value scales, zeroPoints;
    DenseI64ArrayAttr broadcastDims;
    getQuantizationParams(rewriter, op->getLoc(), quantType, scales, zeroPoints,
                          /*outputZeroPointInFp=*/true, broadcastDims);

    Value quantizationMin, quantizationMax;
    getQuantizationStorageInfo(rewriter, op->getLoc(), quantType,
                               quantizationMin, quantizationMax);

    auto resFloatTensorType =
        op.getOperand().getType().clone(rewriter.getF32Type());
    Value resFloat = rewriter.create<chlo::BroadcastDivOp>(
        op->getLoc(), resFloatTensorType, adaptor.getOperand(), scales,
        broadcastDims);
    resFloat = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), resFloatTensorType, resFloat, zeroPoints, broadcastDims);

    resFloat = rewriter.create<stablehlo::ClampOp>(
        op->getLoc(), resFloatTensorType, quantizationMin, resFloat,
        quantizationMax);
    resFloat = rewriter.create<stablehlo::RoundNearestEvenOp>(
        op->getLoc(), resFloatTensorType, resFloat);
    auto resFinalTensorType = resFloatTensorType.clone(
        getQuantStorageType(op.getResult().getType().getElementType()));
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, resFinalTensorType,
                                                      resFloat);
    return success();
  }

  LogicalResult matchAndRewriteRequantize(
      stablehlo::UniformQuantizeOp op,
      stablehlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, QuantType inputQuantType,
      QuantType outputQuantType) const {
    rewriter.replaceOp(
        op,
        requantize(op, adaptor.getOperand(), inputQuantType, outputQuantType,
                   /*outputTensorType=*/
                   cast<TensorType>(op.getResult().getType())
                       .clone(getQuantStorageType(outputQuantType)),
                   rewriter));
    return success();
  }
};

class ConvertUniformDequantizeOp
    : public OpConversionPattern<stablehlo::UniformDequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::UniformDequantizeOp op,
      stablehlo::UniformDequantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto quantType = getQuantType(op.getOperand().getType());
    if (failed(quantType)) {
      return failure();
    }
    Value scales, zeroPoints;
    DenseI64ArrayAttr broadcastDims;
    getQuantizationParams(rewriter, op->getLoc(), *quantType, scales,
                          zeroPoints,
                          /*outputZeroPointInFp=*/false, broadcastDims);

    Value input = adaptor.getOperand();
    // TODO: b/260280919 - Consider avoiding conversion to int32.
    auto resInt32TensorType =
        cast<TensorType>(input.getType()).clone(rewriter.getI32Type());
    Value resInt32 = rewriter.create<stablehlo::ConvertOp>(
        op->getLoc(), resInt32TensorType, input);
    resInt32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), resInt32TensorType, resInt32, zeroPoints, broadcastDims);
    auto resFloatTensorType =
        cast<TensorType>(resInt32.getType()).clone(rewriter.getF32Type());
    Value resFloat = rewriter.create<stablehlo::ConvertOp>(
        op->getLoc(), resFloatTensorType, resInt32);
    resFloat = rewriter.replaceOpWithNewOp<chlo::BroadcastMulOp>(
        op, resFloatTensorType, resFloat, scales, broadcastDims);
    return success();
  }
};

class ConvertUniformQuantizedAddOp
    : public OpConversionPattern<stablehlo::AddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::AddOp op, stablehlo::AddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto lhsQuantType =
        getQuantType(getElementTypeOrSelf(op.getLhs().getType()));
    auto rhsQuantType =
        getQuantType(getElementTypeOrSelf(op.getRhs().getType()));
    auto resQuantType =
        getQuantType(getElementTypeOrSelf(op.getResult().getType()));

    // We only handle cases where lhs, rhs and results all have quantized
    // element type.
    if (failed(lhsQuantType) || failed(rhsQuantType) || failed(resQuantType)) {
      return rewriter.notifyMatchFailure(
          op,
          "AddOp requires the quantized element type for all operands and "
          "results");
    }

    if (isPerAxisType(*lhsQuantType) || isPerAxisType(*rhsQuantType) ||
        isPerAxisType(*resQuantType)) {
      // Handle Per-Axis Quantized Types. We only support lhs/rhs/result with
      // exact same per-axis quantized types with I32 storage type.
      if (!isPerAxisType(*lhsQuantType) || !isPerAxisType(*rhsQuantType) ||
          !isPerAxisType(*resQuantType) ||
          getPerAxisType(*lhsQuantType) != getPerAxisType(*rhsQuantType) ||
          getPerAxisType(*lhsQuantType) != getPerAxisType(*resQuantType)) {
        return rewriter.notifyMatchFailure(
            op,
            "Per-axis quantized AddOp requires the same quantized element "
            "type for all operands and results");
      }
      if (!getPerAxisType(*lhsQuantType).getStorageType().isInteger(32)) {
        // For server-side StableHLO Quantization, add is quantized only when
        // fused with conv/dot ops, whose output must be i32.
        return rewriter.notifyMatchFailure(
            op, "Per-axis quantized AddOp requires i32 storage type");
      }
      return matchAndRewritePerAxis(op, adaptor, rewriter,
                                    getPerAxisType(*lhsQuantType));
    }

    // TODO: b/260280919 - Consider avoiding conversion to int32.
    auto resInt32TensorType =
        op.getResult().getType().clone(rewriter.getI32Type());

    // When lhs, rhs and result have different scale and zps, requantize them to
    // be the same as the result.
    // TODO: b/260280919 - Consider avoiding conversion to int32.
    Value lhs = adaptor.getLhs();
    Value lhsInt32Tensor = requantize(op, lhs, *lhsQuantType, *resQuantType,
                                      resInt32TensorType, rewriter);

    Value rhs = adaptor.getRhs();
    Value rhsInt32Tensor = requantize(op, rhs, *rhsQuantType, *resQuantType,
                                      resInt32TensorType, rewriter);

    Value zeroPoint = rewriter.create<stablehlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          getPerTensorType(*resQuantType).getZeroPoint())));

    // Now the lhs and rhs have been coverted to the same scale and zps.
    // Given:
    // lhs_fp = (lhs_quant - zp) * scale
    // rhs_fp = (rhs_quant - zp) * scale
    // res_fp = lhs_fp + rhs_fp
    //        = ((lhs_quant + rhs_quant - zp) - zp) * scale
    // res_quant = res_fp / scale + zp
    //           = lhs_quant + rhs_quant - zp
    // The following add the inputs and then substract by zero point.
    Value addResult = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), resInt32TensorType, lhsInt32Tensor, rhsInt32Tensor,
        nullptr);
    Value resInt32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), resInt32TensorType, addResult, zeroPoint, nullptr);

    if (getQuantStorageType(*resQuantType).isInteger(32)) {
      // For i32, clamping is not needed.
      rewriter.replaceOp(op, resInt32);
    } else {
      // Clamp results by [quantizationMin, quantizationMax] when storage type
      // is not i32.
      Value resultQuantizationMin = rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(),
          rewriter.getI32IntegerAttr(static_cast<int32_t>(
              getPerTensorType(*resQuantType).getStorageTypeMin())));
      Value resultQuantizationMax = rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(),
          rewriter.getI32IntegerAttr(static_cast<int32_t>(
              getPerTensorType(*resQuantType).getStorageTypeMax())));
      resInt32 = rewriter.create<stablehlo::ClampOp>(
          op->getLoc(), resInt32TensorType, resultQuantizationMin, resInt32,
          resultQuantizationMax);
      // Convert results back to result storage type.
      auto resFinalTensorType =
          resInt32TensorType.clone(getQuantStorageType(*resQuantType));
      rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, resFinalTensorType,
                                                        resInt32);
    }

    return success();
  }

  LogicalResult matchAndRewritePerAxis(
      stablehlo::AddOp op, stablehlo::AddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      quant::UniformQuantizedPerAxisType quantType) const {
    // We assume lhs/rhs/result have the same quantized type with i32 storage.
    Value addResult = rewriter.create<stablehlo::AddOp>(
        op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
    // Add zp contribution if it is non-zero for any axis.
    if (llvm::any_of(quantType.getZeroPoints(),
                     [](int64_t zp) { return zp != 0; })) {
      SmallVector<int32_t> zpsVec(quantType.getZeroPoints().begin(),
                                  quantType.getZeroPoints().end());
      Value zps = rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(),
          DenseIntElementsAttr::get(
              RankedTensorType::get({static_cast<int64_t>(zpsVec.size())},
                                    rewriter.getI32Type()),
              zpsVec));
      addResult = rewriter.create<chlo::BroadcastSubOp>(
          op->getLoc(), addResult, zps,
          rewriter.getDenseI64ArrayAttr(
              {static_cast<int64_t>(quantType.getQuantizedDimension())}));
    }
    rewriter.replaceOp(op, addResult);
    return success();
  }
};

// This is a convenient struct for holding dimension numbers for dot-like ops
// including DotGeneral and Convolution. So that we can share code for all
// dot-like ops.
// For Convolution, only NHWC format is supported.
// For DotGeneral, there is no contracting dims. The batching and contracting
// dimensions are defined in
// https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general.
struct DotLikeDimensionNumbers {
  SmallVector<int64_t> lhsBatchingDims;
  SmallVector<int64_t> lhsSpatialDims;
  SmallVector<int64_t> lhsContractingDims;
  SmallVector<int64_t> rhsBatchingDims;
  SmallVector<int64_t> rhsSpatialDims;
  SmallVector<int64_t> rhsContractingDims;
};

// Checks if zero points of the given quantized type are zero.
bool isZeroPointZero(QuantType type) {
  if (isPerTensorType(type)) {
    return getPerTensorType(type).getZeroPoint() == 0;
  }
  if (isPerAxisType(type)) {
    ArrayRef<int64_t> zeroPoints = getPerAxisType(type).getZeroPoints();
    return llvm::all_of(zeroPoints, [](int64_t zp) { return zp == 0; });
  }
  return false;
}

// A shared matchAndRewrite implementation for dot-like hybrid quantized
// operators. Hybrid ops are currently only interpreted as weight-only
// quantization ops, this might change in the future.
//
// All attrs of the original op are preserved after the conversion.
template <typename OpType, typename OpAdaptorType>
LogicalResult matchAndRewriteDotLikeHybridOp(
    OpType &op, OpAdaptorType &adaptor, ConversionPatternRewriter &rewriter) {
  // For dot like hybrid ops, lhs is float type, rhs is uniform
  // quantized type and result is float type.
  // For weight-only quantization:
  // result = hybridOp(lhs, dequant(rhs))
  Value lhsFloat32Tensor = adaptor.getLhs();
  // Insert optimization_barrier to prevent constant folding of dequantize +
  // quantized weights.
  auto barrier = rewriter.create<stablehlo::OptimizationBarrierOp>(
      op->getLoc(), adaptor.getRhs());
  Operation::result_range resultRange = barrier.getResults();
  Value rhs = resultRange.front();
  FailureOr<QuantType> rhsElementQuantType =
      getQuantType(op.getRhs().getType());
  if (failed(rhsElementQuantType)) {
    return failure();
  }
  auto resFloat32TensorType = cast<TensorType>(op.getResult().getType());
  auto rhsFloat32TensorType =
      cast<TensorType>(op.getRhs().getType()).clone(rewriter.getF32Type());

  // Get scales and zero points for rhs.
  Value rhsScale, rhsZeroPoint;
  DenseI64ArrayAttr broadcastDims;
  getQuantizationParams(rewriter, op->getLoc(), *rhsElementQuantType, rhsScale,
                        rhsZeroPoint,
                        /*outputZeroPointInFp=*/true, broadcastDims);

  // Dequantize rhs_float32_tensor.
  Value rhsFloat32Tensor = rewriter.create<stablehlo::ConvertOp>(
      op->getLoc(), rhsFloat32TensorType, rhs);

  // Subtract zero points only when it is not zero.
  if (!isZeroPointZero(*rhsElementQuantType)) {
    rhsFloat32Tensor = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), rhsFloat32TensorType, rhsFloat32Tensor, rhsZeroPoint,
        broadcastDims);
  }
  rhsFloat32Tensor = rewriter.create<chlo::BroadcastMulOp>(
      op->getLoc(), rhsFloat32TensorType, rhsFloat32Tensor, rhsScale,
      broadcastDims);

  // Execute conversion target op.
  SmallVector<Value, 2> operands{lhsFloat32Tensor, rhsFloat32Tensor};
  rewriter.replaceOpWithNewOp<OpType>(op, resFloat32TensorType, operands,
                                      op->getAttrs());
  return success();
}

Value createZeroPointPartialOffset(OpBuilder &builder, Location loc,
                                   Value tensor, const int64_t otherTensorZp,
                                   SmallVector<int64_t> reductionDims) {
  // This function calculates part of the zero-point-offset by using
  // stablehlo::Reduce to sum over the contracting dims of the tensor, and then
  // multiply by zp of the other tensor.
  auto outputElementType = builder.getI32Type();

  // Calculate the output tensor shape. This is input tensor dims minus
  // contracting dims.
  auto rankedTensor = cast<RankedTensorType>(tensor.getType());
  SmallVector<int64_t> outputDims;
  for (int64_t i = 0; i < rankedTensor.getRank(); ++i) {
    if (llvm::count(reductionDims, i) == 0) {
      outputDims.push_back(rankedTensor.getDimSize(i));
    }
  }

  // Convert input tensor to output type since stablehlo::Reduce only supports
  // same element type for input/output.
  tensor = builder.create<stablehlo::ConvertOp>(
      loc, cast<TensorType>(tensor.getType()).clone(outputElementType), tensor);
  auto reducerTensorType = RankedTensorType::get({}, outputElementType);

  // Initial value for reduced tensor. This is set 0.
  Value initValues = builder.create<stablehlo::ConstantOp>(
      loc, DenseIntElementsAttr::get(reducerTensorType, {0}));
  stablehlo::ReduceOp reduce = builder.create<stablehlo::ReduceOp>(
      loc, RankedTensorType::get(outputDims, outputElementType), tensor,
      initValues, reductionDims);
  // Define reducer function to compute sum.
  Region &region = reduce.getBody();
  Block &block = region.emplaceBlock();
  block.addArgument(reducerTensorType, loc);
  block.addArgument(reducerTensorType, loc);
  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);
    Value sum =
        builder.create<stablehlo::AddOp>(loc, *firstArgument, *secondArgument);
    builder.create<stablehlo::ReturnOp>(loc, sum);
  }
  Value zp = builder.create<stablehlo::ConstantOp>(
      loc, builder.getI32IntegerAttr(otherTensorZp));
  Value mulOp = builder.create<chlo::BroadcastMulOp>(loc, reduce.getResult(0),
                                                     zp, nullptr);
  return mulOp;
}

Value getDimValue(OpBuilder &builder, Location loc, Value tensor,
                  ShapedType tensorShape, int64_t idx) {
  if (tensorShape.isDynamicDim(idx)) {
    // Get dynamic dim using GetDimensionSizeOp and convert result from <i32> to
    // <1xi64>.
    Value dynamicDim = builder.create<stablehlo::GetDimensionSizeOp>(
        loc, tensor, builder.getI64IntegerAttr(idx));
    dynamicDim = builder.create<stablehlo::ConvertOp>(
        loc, RankedTensorType::get(ArrayRef<int64_t>{}, builder.getI64Type()),
        dynamicDim);
    return builder.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({1}, builder.getI64Type()), dynamicDim);
  }
  return builder.create<stablehlo::ConstantOp>(
      loc, DenseIntElementsAttr::get(
               RankedTensorType::get({1}, builder.getI64Type()),
               {tensorShape.getDimSize(idx)}));
}

Value calculateDynamicOutputDims(OpBuilder &builder, Location loc, Value output,
                                 ShapedType outputTensorType) {
  // Calculate each output tensor dim and concatenate into a 1D tensor.
  SmallVector<Value> outputDims;
  for (int64_t i = 0; i < outputTensorType.getRank(); ++i) {
    outputDims.push_back(
        getDimValue(builder, loc, output, outputTensorType, i));
  }
  return builder.create<stablehlo::ConcatenateOp>(loc, outputDims,
                                                  builder.getI64IntegerAttr(0));
}

Value broadcastZpContribution(OpBuilder &builder, Location loc,
                              Value zpContribution,
                              ArrayRef<int64_t> reductionDims,
                              ArrayRef<int64_t> batchingDims,
                              int64_t nonBatchingStartingIdx, Value output,
                              TensorType outputTensorType,
                              Value &outputDimsValue) {
  // This function calculates the dims for broadcasting from the
  // zero-point-offset tensor to the final output tensor, and then do the
  // broadcast.
  auto zpContributionRank =
      cast<ShapedType>(zpContribution.getType()).getRank();
  SmallVector<int64_t> broadcastDims;
  broadcastDims.resize(zpContributionRank, 0);
  // Result tensor will have batching dims first, then LHS result dims, then
  // RHS result dims. So non-batching result dims index doesn't start from 0.
  // The arg non_batching_starting_idx is used to distinguish LHS and RHS.
  int64_t resultBatchingIdx = 0;
  int64_t resultNonBatchingIdx = nonBatchingStartingIdx;
  for (int64_t idx = 0, originalIdx = 0; idx < zpContributionRank;
       ++idx, ++originalIdx) {
    // zp_contribution has removed contracting/spatial dims from the tensor
    // after reduction. The following recovers the index in the original tensor.
    while (llvm::count(reductionDims, originalIdx) != 0) {
      originalIdx++;
    }
    if (llvm::count(batchingDims, originalIdx) == 0) {
      broadcastDims[idx] = resultNonBatchingIdx++;
    } else {
      broadcastDims[idx] = resultBatchingIdx++;
    }
  }
  // Use broadcast_in_dim or dyanmic_broadcast_in_dim based on output shape
  // dynamism.
  if (cast<ShapedType>(outputTensorType).hasStaticShape()) {
    zpContribution = builder.create<stablehlo::BroadcastInDimOp>(
        loc, outputTensorType, zpContribution, broadcastDims);
  } else {
    if (!outputDimsValue) {
      outputDimsValue =
          calculateDynamicOutputDims(builder, loc, output, outputTensorType);
    }
    zpContribution = builder.create<stablehlo::DynamicBroadcastInDimOp>(
        loc, outputTensorType, zpContribution, outputDimsValue,
        builder.getDenseI64ArrayAttr(broadcastDims));
  }
  return zpContribution;
}

Value calculateZeroPointOffset(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs, Value output, int64_t lhsZp,
                               int64_t rhsZp, TensorType outputTensorType,
                               const DotLikeDimensionNumbers &dims) {
  ShapedType lhsShape = cast<ShapedType>(lhs.getType());
  ShapedType rhsShape = cast<ShapedType>(rhs.getType());
  Value result = nullptr;
  Value outputDimsValue = nullptr;
  // Calculate LHS contribution when RHS zp is non-zero.
  if (rhsZp != 0) {
    SmallVector<int64_t> reductionDims = to_vector(llvm::concat<const int64_t>(
        dims.lhsSpatialDims, dims.lhsContractingDims));
    Value lhsZpContribution =
        createZeroPointPartialOffset(builder, loc, lhs, rhsZp, reductionDims);
    // Broadcast lhs ZP contribution to result tensor shape.
    lhsZpContribution = broadcastZpContribution(
        builder, loc, lhsZpContribution, reductionDims, dims.lhsBatchingDims,
        dims.lhsBatchingDims.size(), output, outputTensorType, outputDimsValue);
    result = lhsZpContribution;
  }
  // Calculate RHS contribution when LHS zp is non-zero.
  if (lhsZp != 0) {
    SmallVector<int64_t> reductionDims = to_vector(llvm::concat<const int64_t>(
        dims.rhsSpatialDims, dims.rhsContractingDims));
    Value rhsZpContribution =
        createZeroPointPartialOffset(builder, loc, rhs, lhsZp, reductionDims);
    // Broadcast rhs ZP contribution to result tensor shape.
    rhsZpContribution = broadcastZpContribution(
        builder, loc, rhsZpContribution, reductionDims, dims.rhsBatchingDims,
        lhsShape.getRank() - dims.lhsContractingDims.size(), output,
        outputTensorType, outputDimsValue);
    if (result) {
      result = builder.create<stablehlo::AddOp>(loc, result, rhsZpContribution);
    } else {
      result = rhsZpContribution;
    }
  }

  if (lhsZp != 0 && rhsZp != 0) {
    // Contributions from LHS_ZP * RHS_ZP.
    // This is multiplied by the product of all contracting dimensions.
    int32_t contractingDimTotalInt = 1;
    bool hasDynamicContractingDim = false;
    Value dynamicContractingDimTotal = builder.create<stablehlo::ConstantOp>(
        loc, builder.getI32IntegerAttr(static_cast<int32_t>(1)));
    // Calculate the product for static/dynamic dims separately.
    for (int64_t rhsIdx : llvm::concat<const int64_t>(
             dims.rhsSpatialDims, dims.rhsContractingDims)) {
      if (rhsShape.isDynamicDim(rhsIdx)) {
        hasDynamicContractingDim = true;
        auto dim = builder.create<stablehlo::GetDimensionSizeOp>(
            loc, rhs, builder.getI64IntegerAttr(rhsIdx));
        dynamicContractingDimTotal = builder.create<stablehlo::MulOp>(
            loc, dynamicContractingDimTotal, dim);
      } else {
        contractingDimTotalInt *= rhsShape.getDimSize(rhsIdx);
      }
    }
    Value zpOffsetValue = builder.create<stablehlo::ConstantOp>(
        loc, builder.getI32IntegerAttr(static_cast<int32_t>(lhsZp) *
                                       static_cast<int32_t>(rhsZp) *
                                       contractingDimTotalInt));
    // Multiply the static dims contribution by the dynamic one if needed.
    if (hasDynamicContractingDim) {
      zpOffsetValue = builder.create<stablehlo::MulOp>(
          loc, zpOffsetValue, dynamicContractingDimTotal);
    }
    result = builder.create<chlo::BroadcastSubOp>(loc, result, zpOffsetValue,
                                                  nullptr);
  }
  return result;
}

// Generic function to create DotGeneral kernel for Dot/DotGeneral ops.
template <typename DotLikeOp>
Value createDotLikeKernel(OpBuilder &builder, Location loc, DotLikeOp,
                          Type resultType, Value &lhs, Value &rhs,
                          ArrayRef<NamedAttribute> attrs) {
  return builder.create<stablehlo::DotGeneralOp>(
      loc, resultType, ArrayRef<Value>{lhs, rhs}, attrs);
}

// Template specialization for Convolution op.
// This function may pad LHS if needed. If so, lhs is updated in place.
template <>
Value createDotLikeKernel<stablehlo::ConvolutionOp>(
    OpBuilder &builder, Location loc, stablehlo::ConvolutionOp op,
    Type resultType, Value &lhs, Value &rhs, ArrayRef<NamedAttribute> attrs) {
  // We only handle the case where RHS zp is zero.
  // Explicitly pad LHS with zp and update LHS value.
  SmallVector<NamedAttribute> newAttrs(attrs);
  if (op.getPadding().has_value() &&
      llvm::any_of(op.getPaddingAttr().getValues<int64_t>(),
                   [](int64_t x) { return x != 0; })) {
    auto originalPadding = op.getPaddingAttr().getValues<int64_t>();

    Value zp = builder.create<stablehlo::ConstantOp>(
        loc, DenseIntElementsAttr::get(
                 RankedTensorType::get({}, builder.getI8Type()),
                 {static_cast<int8_t>(
                     getPerTensorType(op.getLhs().getType()).getZeroPoint())}));
    // Convert Padding attributes from stablehlo::Convolution to stablehlo::Pad.
    // Note that Padding is applied for spatial dimensions [1...rank-1) only for
    // stablehlo::Convolution. But stablehlo::Pad require those for all
    // dimensions. Hence we add 0 to the beginning and end of the padding
    // vectors.
    int64_t rank = cast<TensorType>(lhs.getType()).getRank();
    SmallVector<int64_t> paddingLow(rank, 0), paddingHigh(rank, 0),
        paddingInterior(rank, 0);
    for (int64_t i = 1; i < rank - 1; ++i) {
      paddingLow[i] = originalPadding[i * 2 - 2];
      paddingHigh[i] = originalPadding[i * 2 - 1];
    }
    lhs = builder.create<stablehlo::PadOp>(loc, lhs, zp, paddingLow,
                                           paddingHigh, paddingInterior);

    // After explicitly padding/dilating LHS, update attributes so that LHS is
    // not padded/dilated again during Convolution.
    for (auto &attr : newAttrs) {
      if (attr.getName().getValue() == "padding") {
        attr.setValue(SplatElementsAttr::get(
            RankedTensorType::get({rank - 2, 2}, builder.getI64Type()),
            builder.getI64IntegerAttr(0)));
      }
    }
  }
  return builder.create<stablehlo::ConvolutionOp>(
      loc, resultType, ArrayRef<Value>{lhs, rhs}, newAttrs);
}

template <typename DotLikeOp, typename DotLikeOpAdaptor>
LogicalResult matchAndRewriteDotLikeOp(DotLikeOp op, DotLikeOpAdaptor adaptor,
                                       ArrayRef<NamedAttribute> attrs,
                                       const DotLikeDimensionNumbers &dims,
                                       ConversionPatternRewriter &rewriter) {
  // Lower Dot/DotGeneral UQ ops to DotGeneral int.
  // Assumes that operands and results are uq types.
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  auto resInt32TensorType =
      op.getResult().getType().clone(rewriter.getI32Type());

  // Dot result
  //   = dot((lhs - zp_l) * scale_l, (rhs - zp_r) * scale_r) / scale_res
  //       + zp_res
  //   = dot(lhs - zp_l, rhs - zp_r) * scale_l * scale_r / scale_res + zp_res
  //   = dot(lhs, rhs) * combined_scale + combined_zp
  // where:
  //   combined_scale = scale_l * scale_r / scale_res
  //   combined_zp = res_zp - zp_offset * combined_scale
  //   zp_offset = zp_l*rhs + zp_r*lhs - zp_l*zp_r
  Value resI32 = createDotLikeKernel(rewriter, op->getLoc(), op,
                                     resInt32TensorType, lhs, rhs, attrs);

  auto lhsElementQuantType = getPerTensorType(op.getLhs().getType());
  auto rhsElementQuantType = dyn_cast<quant::UniformQuantizedType>(
      getElementTypeOrSelf(op.getRhs().getType()));
  auto rhsElementQuantPerAxisType =
      dyn_cast<quant::UniformQuantizedPerAxisType>(
          getElementTypeOrSelf(op.getRhs().getType()));
  auto resElementQuantType = dyn_cast<quant::UniformQuantizedType>(
      getElementTypeOrSelf(op.getResult()));
  auto resElementQuantPerAxisType =
      dyn_cast<quant::UniformQuantizedPerAxisType>(
          getElementTypeOrSelf(op.getResult()));

  // Here we assume LHS must be per-tensor quantized.
  // If RHS is per-axis quantized, it must has 0 zp.
  Value zpOffset = calculateZeroPointOffset(
      rewriter, op->getLoc(), lhs, rhs, resI32,
      lhsElementQuantType.getZeroPoint(),
      (rhsElementQuantType ? rhsElementQuantType.getZeroPoint() : 0),
      resInt32TensorType, dims);

  // For per-axis quantization, we assume that result scales are proportional
  // to rhs scales for each channel axis.
  double combinedScaleFp = rhsElementQuantType
                               ? lhsElementQuantType.getScale() *
                                     rhsElementQuantType.getScale() /
                                     resElementQuantType.getScale()
                               : lhsElementQuantType.getScale() *
                                     rhsElementQuantPerAxisType.getScales()[0] /
                                     resElementQuantPerAxisType.getScales()[0];

  // Multiply dot result and zp_offset by combined_scale only if it is not 1.0.
  if (std::abs(combinedScaleFp - 1.0) > 0.001) {
    Value combinedScale = rewriter.create<stablehlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(combinedScaleFp));

    auto resFloat32TensorType =
        op.getResult().getType().clone(rewriter.getF32Type());
    Value resF32 = rewriter.create<stablehlo::ConvertOp>(
        op->getLoc(), resFloat32TensorType, resI32);
    resF32 = rewriter.create<chlo::BroadcastMulOp>(
        op->getLoc(), resFloat32TensorType, resF32, combinedScale, nullptr);
    resI32 = rewriter.create<stablehlo::ConvertOp>(op->getLoc(),
                                                   resInt32TensorType, resF32);

    // Skip zp_offset if it is 0.
    if (zpOffset) {
      auto zpOffsetFloat32TensorType =
          cast<TensorType>(zpOffset.getType()).clone(rewriter.getF32Type());
      zpOffset = rewriter.create<stablehlo::ConvertOp>(
          op->getLoc(), zpOffsetFloat32TensorType, zpOffset);
      zpOffset = rewriter.create<chlo::BroadcastMulOp>(
          op->getLoc(), zpOffsetFloat32TensorType, zpOffset, combinedScale,
          nullptr);
      zpOffset = rewriter.create<stablehlo::ConvertOp>(
          op->getLoc(), zpOffsetFloat32TensorType.clone(rewriter.getI32Type()),
          zpOffset);
    }
  }

  // If result is per-axis quantized, it must has 0 zp.
  Value combinedZp = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(),
      rewriter.getI32IntegerAttr(
          resElementQuantType ? resElementQuantType.getZeroPoint() : 0));
  if (zpOffset) {
    combinedZp = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), resInt32TensorType, combinedZp, zpOffset, nullptr);
  }
  Value zpAdded = rewriter.create<chlo::BroadcastAddOp>(
      op->getLoc(), resInt32TensorType, resI32, combinedZp, nullptr);

  // Convert results back to result storage type.
  auto resQuantType = getQuantType(getElementTypeOrSelf(op.getResult()));
  auto resFinalTensorType =
      resInt32TensorType.clone(getQuantStorageType(*resQuantType));
  rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, resFinalTensorType,
                                                    zpAdded);
  return success();
}

template <typename DotLikeOp>
FailureOr<bool> isDotLikeOpHybrid(DotLikeOp op,
                                  ConversionPatternRewriter &rewriter) {
  // Checks whether a dot-like op is hybrid by looking at input/output types.
  // Returns failure() when the type is not supported.
  bool isLhsQuant = isa<quant::UniformQuantizedType>(
      getElementTypeOrSelf(op.getLhs().getType()));
  bool isLhsQuantPerAxis = isa<quant::UniformQuantizedPerAxisType>(
      getElementTypeOrSelf(op.getLhs().getType()));
  bool isRhsQuant = isa<quant::UniformQuantizedType>(
      getElementTypeOrSelf(op.getRhs().getType()));
  bool isRhsQuantPerAxis = isa<quant::UniformQuantizedPerAxisType>(
      getElementTypeOrSelf(op.getRhs().getType()));
  bool isResQuant =
      isa<quant::UniformQuantizedType>(getElementTypeOrSelf(op.getResult()));
  bool isResQuantPerAxis = isa<quant::UniformQuantizedPerAxisType>(
      getElementTypeOrSelf(op.getResult()));

  if (isLhsQuant && ((isRhsQuant && isResQuant) ||
                     (isRhsQuantPerAxis && isResQuantPerAxis))) {
    // For quantized ops, RHS and result must be both per-axis quantized or
    // both per-tensor quantized.
    return false;
  }
  if (!isLhsQuant && !isLhsQuantPerAxis && (isRhsQuant || isRhsQuantPerAxis) &&
      !isResQuant && !isResQuantPerAxis) {
    return true;
  }
  return rewriter.notifyMatchFailure(
      op, "Invalid input/output type for Dot/Convolution op");
}

class ConvertUniformQuantizedDotOp
    : public OpConversionPattern<stablehlo::DotOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::DotOp op, stablehlo::DotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto isHybrid = isDotLikeOpHybrid(op, rewriter);
    if (failed(isHybrid)) {
      return failure();
    }
    if (*isHybrid) {
      return matchAndRewriteDotLikeHybridOp(op, adaptor, rewriter);
    }  // DotOp is a special case of DotGeneralOp, where LHS and RHS are both
    // rank-2 tensors and have contracting dims of 1 and 0 respectively.
    auto dims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{1},
        /*rhsContractingDimensions=*/{0});
    SmallVector<NamedAttribute> attrs(op->getAttrs());
    attrs.push_back(
        {StringAttr::get(rewriter.getContext(), "dot_dimension_numbers"),
         dims});
    return matchAndRewriteDotLikeOp(
        op, adaptor, attrs,
        DotLikeDimensionNumbers{/*lhs_batching_dims=*/{},
                                /*lhs_spatial_dims=*/{},
                                /*lhs_contracting_dims=*/{1},
                                /*rhs_batching_dims=*/{},
                                /*rhs_spatial_dims=*/{},
                                /*rhs_contracting_dims=*/{0}},
        rewriter);
  }
};

class ConvertUniformQuantizedDotGeneralOp
    : public OpConversionPattern<stablehlo::DotGeneralOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::DotGeneralOp op, stablehlo::DotGeneralOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto isHybrid = isDotLikeOpHybrid(op, rewriter);
    if (failed(isHybrid)) {
      return failure();
    }
    if (*isHybrid) {
      return matchAndRewriteDotLikeHybridOp(op, adaptor, rewriter);
    }
    return matchAndRewriteDotLikeOp(
        op, adaptor, op->getAttrs(),
        DotLikeDimensionNumbers{
            to_vector(op.getDotDimensionNumbers().getLhsBatchingDimensions()),
            /*lhs_spatial_dims=*/{},
            to_vector(
                op.getDotDimensionNumbers().getLhsContractingDimensions()),
            to_vector(op.getDotDimensionNumbers().getRhsBatchingDimensions()),
            /*rhs_spatial_dims=*/{},
            to_vector(
                op.getDotDimensionNumbers().getRhsContractingDimensions())},
        rewriter);
  }
};

bool isConvNhwc(const stablehlo::ConvDimensionNumbersAttr &dims) {
  return dims.getInputBatchDimension() == 0 &&
         dims.getInputFeatureDimension() == 3 &&
         dims.getInputSpatialDimensions().size() == 2 &&
         dims.getInputSpatialDimensions()[0] == 1 &&
         dims.getInputSpatialDimensions()[1] == 2 &&
         dims.getKernelInputFeatureDimension() == 2 &&
         dims.getKernelOutputFeatureDimension() == 3 &&
         dims.getKernelSpatialDimensions().size() == 2 &&
         dims.getKernelSpatialDimensions()[0] == 0 &&
         dims.getKernelSpatialDimensions()[1] == 1 &&
         dims.getOutputBatchDimension() == 0 &&
         dims.getOutputFeatureDimension() == 3 &&
         dims.getOutputSpatialDimensions().size() == 2 &&
         dims.getOutputSpatialDimensions()[0] == 1 &&
         dims.getOutputSpatialDimensions()[1] == 2;
}

bool isConvNDHWC(const stablehlo::ConvDimensionNumbersAttr &dims) {
  return dims.getInputBatchDimension() == 0 &&
         dims.getInputFeatureDimension() == 4 &&
         dims.getInputSpatialDimensions().size() == 3 &&
         dims.getInputSpatialDimensions()[0] == 1 &&
         dims.getInputSpatialDimensions()[1] == 2 &&
         dims.getInputSpatialDimensions()[2] == 3 &&
         dims.getKernelInputFeatureDimension() == 3 &&
         dims.getKernelOutputFeatureDimension() == 4 &&
         dims.getKernelSpatialDimensions().size() == 3 &&
         dims.getKernelSpatialDimensions()[0] == 0 &&
         dims.getKernelSpatialDimensions()[1] == 1 &&
         dims.getKernelSpatialDimensions()[2] == 2 &&
         dims.getOutputBatchDimension() == 0 &&
         dims.getOutputFeatureDimension() == 4 &&
         dims.getOutputSpatialDimensions().size() == 3 &&
         dims.getOutputSpatialDimensions()[0] == 1 &&
         dims.getOutputSpatialDimensions()[1] == 2 &&
         dims.getOutputSpatialDimensions()[2] == 3;
}

FailureOr<DotLikeDimensionNumbers> verifyAndConstructDims(
    stablehlo::ConvolutionOp op, ConversionPatternRewriter &rewriter) {
  // RHS (weight) must have zero zp.
  // Here assumes RHS/result must be both per-tensor or both per-axis
  // quantized.
  auto failedOr = getQuantType(op.getRhs().getType());
  if (failed(failedOr)) {
    return failure();
  }
  QuantType rhsElementQuantType = *failedOr;
  bool isRhsQuantPerTensor = isPerTensorType(rhsElementQuantType);

  if (isRhsQuantPerTensor
          ? getPerTensorType(rhsElementQuantType).getZeroPoint() != 0
          : llvm::any_of(
                llvm::concat<const int64_t>(
                    getPerAxisType(rhsElementQuantType).getZeroPoints(),
                    getPerAxisType(op.getType()).getZeroPoints()),
                [](int64_t zp) { return zp != 0; })) {
    return rewriter.notifyMatchFailure(op,
                                       "RHS/result UQ type must have zero zp.");
  }
  // For per-axis quantization, RHS quantized axis must be out channel axis.
  if (!isRhsQuantPerTensor &&
      (getPerAxisType(rhsElementQuantType).getQuantizedDimension() !=
       cast<TensorType>(op.getRhs().getType()).getRank() - 1)) {
    return rewriter.notifyMatchFailure(
        op, "Conv quantized axis must be out channel axis");
  }
  // For per-axis quantization, ratio between RHS and Result scales must be
  // the same for each channel.
  if (!isRhsQuantPerTensor) {
    auto resElementQuantPerAxisType = getPerAxisType(op.getType());
    SmallVector<double> scaleRatios(
        resElementQuantPerAxisType.getScales().size());
    for (size_t i = 0; i < scaleRatios.size(); ++i) {
      scaleRatios[i] = resElementQuantPerAxisType.getScales()[i] /
                       getPerAxisType(rhsElementQuantType).getScales()[i];
      auto diff = (scaleRatios[i] - scaleRatios[0]) / scaleRatios[0];
      // Check all ratios within a threshold.
      if (std::abs(diff) > 0.001) {
        return rewriter.notifyMatchFailure(
            op,
            "Per-axis quantizated Conv must have same RHS/Result scale "
            "ratio for each channel");
      }
    }
  }
  // lhs_dilation must not exist.
  if (op.getLhsDilation().has_value() &&
      llvm::any_of(*op.getLhsDilation(),
                   [](int64_t dilate) { return dilate != 1; })) {
    return rewriter.notifyMatchFailure(op, "lhs_dilation must be 1.");
  }

  // We only support NHWC Conv2D and NDHWC Conv3D.
  auto dims = op.getDimensionNumbers();
  if (isConvNhwc(dims)) {
    // 2D Convolution.
    return DotLikeDimensionNumbers{/*lhs_batching_dims=*/{0},
                                   /*lhs_spatial_dims=*/{1, 2},
                                   /*lhs_contracting_dims=*/{3},
                                   /*rhs_batching_dims=*/{},
                                   /*rhs_spatial_dims=*/{0, 1},
                                   /*rhs_contracting_dims=*/{2}};
  }
  if (isConvNDHWC(dims)) {
    // 3D Convolution.
    return DotLikeDimensionNumbers{/*lhs_batching_dims=*/{0},
                                   /*lhs_spatial_dims=*/{1, 2, 3},
                                   /*lhs_contracting_dims=*/{4},
                                   /*rhs_batching_dims=*/{},
                                   /*rhs_spatial_dims=*/{0, 1, 2},
                                   /*rhs_contracting_dims=*/{3}};
  }
  return rewriter.notifyMatchFailure(op,
                                     "Convolution data format must be NHWC.");
}

class ConvertUniformQuantizedConvolutionOp
    : public OpConversionPattern<stablehlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::ConvolutionOp op, stablehlo::ConvolutionOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto isHybrid = isDotLikeOpHybrid(op, rewriter);
    if (failed(isHybrid)) {
      return failure();
    }
    if (*isHybrid) {
      return matchAndRewriteDotLikeHybridOp(op, adaptor, rewriter);
    }
    auto dims = verifyAndConstructDims(op, rewriter);
    if (failed(dims)) return failure();
    return matchAndRewriteDotLikeOp(op, adaptor, op->getAttrs(), *dims,
                                    rewriter);
  }
};

// This pattern lowers a generic stablehlo op for uq->int.
// This pattern essentially just performs type change, with no algorithm change.
// TODO: b/310685906 - Add operand/result type validations.
class ConvertGenericOp : public ConversionPattern {
 public:
  explicit ConvertGenericOp(MLIRContext *ctx, TypeConverter &converter,
                            PatternBenefit benefit)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), benefit, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // This pattern only handle selected ops.
    if (!isa<stablehlo::BitcastConvertOp, stablehlo::BroadcastInDimOp,
             stablehlo::ConcatenateOp, stablehlo::ConstantOp,
             stablehlo::DynamicReshapeOp, stablehlo::DynamicSliceOp,
             stablehlo::GatherOp, stablehlo::MaxOp, stablehlo::MinOp,
             stablehlo::PadOp, stablehlo::ReduceWindowOp, stablehlo::ReshapeOp,
             stablehlo::ReturnOp, stablehlo::SelectOp, stablehlo::SliceOp,
             stablehlo::TransposeOp, stablehlo::GetDimensionSizeOp,
             stablehlo::DynamicBroadcastInDimOp>(op)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported op for performing type change");
    }

    if (isa<stablehlo::MinOp, stablehlo::MaxOp>(op)) {
      // Min/max only support per-tensor quantization.
      auto lhsType = getPerTensorType(op->getOperandTypes()[0]);
      auto rhsType = getPerTensorType(op->getOperandTypes()[1]);
      auto resultType = getPerTensorType(op->getResultTypes()[0]);
      if (lhsType != rhsType || lhsType != resultType) {
        return rewriter.notifyMatchFailure(
            op, op->getName().getStringRef() +
                    " with different quantization parameters for operands and"
                    " results is not supported.");
      }
    }

    // Determine new result type: use storage type for uq types; use original
    // type otherwise.
    SmallVector<Type, 4> newResultTypes;
    for (auto resultType : op->getResultTypes()) {
      newResultTypes.push_back(getQuantStorageType(resultType));
    }

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResultTypes, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      auto newRegion = std::make_unique<Region>(op);
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
      if (failed(rewriter.convertRegionTypes(newRegion.get(),
                                             *getTypeConverter()))) {
        return failure();
      }
      state.addRegion(std::move(newRegion));
    }
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// TypeConverter for converting UQ type to int type.
class UniformQuantizedToIntTypeConverter : public TypeConverter {
 public:
  UniformQuantizedToIntTypeConverter() {
    addConversion([](Type type) -> Type { return getQuantStorageType(type); });
  }
};

}  // namespace

#define GEN_PASS_DEF_STABLEHLOLEGALIZEQUANTTOMATHPASS
#include "stablehlo/transforms/Passes.h.inc"

class StablehloLegalizeQuantToMathPass
    : public impl::StablehloLegalizeQuantToMathPassBase<
          StablehloLegalizeQuantToMathPass> {
 public:
  // Performs conversion of stablehlo quant ops to primitive ops.
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    // Populate stablehlo quant ops conversion patterns.
    patterns.add<ConvertUniformQuantizeOp, ConvertUniformDequantizeOp,
                 ConvertUniformQuantizedAddOp, ConvertUniformQuantizedDotOp,
                 ConvertUniformQuantizedDotGeneralOp,
                 ConvertUniformQuantizedConvolutionOp>(context, /*benefit=*/10);

    // Populate stablehlo quant-op to dq-op-q patterns as fallback.
    populateStablehloLegalizeQuantizedOpToQDQPatterns(&patterns, context,
                                                      /*benefit=*/1);

    // uq->int convert patterns for func.func, func.return and generic ops.
    UniformQuantizedToIntTypeConverter converter;
    patterns.add<ConvertGenericOp>(context, converter, /*benefit=*/10);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    ConversionTarget target(*op->getContext());
    target.addIllegalDialect<quant::QuantDialect>();
    auto isLegal = [&converter](Operation *op) {
      return converter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<stablehlo::StablehloDialect>(isLegal);
    target.addDynamicallyLegalDialect<chlo::ChloDialect>(isLegal);
    target.addDynamicallyLegalDialect<func::FuncDialect>(
        [&converter](Operation *op) {
          if (auto func = dyn_cast<func::FuncOp>(op)) {
            return converter.isSignatureLegal(func.getFunctionType());
          }
          return converter.isLegal(op);
        });

    LogicalResult result =
        applyPartialConversion(op, target, std::move(patterns));
    if (failed(result)) {
      signalPassFailure();
    }
  }
};

}  // namespace mlir::stablehlo
