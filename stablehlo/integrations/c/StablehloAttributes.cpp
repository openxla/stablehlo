/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
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

#include "stablehlo/integrations/c/StablehloAttributes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloScatterDimensionNumbersGet(
    MlirContext ctx, intptr_t nUpdateWindowDims,
    const int64_t *updateWindowDims, intptr_t nInsertedWindowDims,
    const int64_t *insertedWindowDims, intptr_t nScatteredDimsToOperandDims,
    const int64_t *scatteredDimsToOperandDims, int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      unwrap(ctx), llvm::ArrayRef(updateWindowDims, nUpdateWindowDims),
      llvm::ArrayRef(insertedWindowDims, nInsertedWindowDims),
      llvm::ArrayRef(scatteredDimsToOperandDims, nScatteredDimsToOperandDims),
      indexVectorDim));
}

bool stablehloAttributeIsAScatterDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ScatterDimensionNumbersAttr>();
}

intptr_t stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getUpdateWindowDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getInsertedWindowDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getScatterDimsToOperandDims()[pos];
}

int64_t stablehloDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ScatterDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::GatherDimensionNumbersAttr::get(
      unwrap(ctx), llvm::ArrayRef(offsetDims, nOffsetDims),
      llvm::ArrayRef(collapsedSliceDims, nCollapsedSliceDims),
      llvm::ArrayRef(startIndexMap, nStartIndexMap), indexVectorDim));
}

bool stablehloAttributeIsAGatherDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::GatherDimensionNumbersAttr>();
}

intptr_t stablehloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetOffsetDimsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getOffsetDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getCollapsedSliceDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetStartIndexMapSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetStartIndexMapElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getStartIndexMap()[pos];
}

int64_t stablehloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::GatherDimensionNumbersAttr>()
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// DotDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloDotDimensionNumbersGet(
    MlirContext ctx, intptr_t nLhsBatchingDimensions,
    const int64_t *lhsBatchingDimensions, intptr_t nRhsBatchingDimensions,
    const int64_t *rhsBatchingDimensions, intptr_t nLhsContractingDimensions,
    const int64_t *lhsContractingDimensions, intptr_t nRhsContractingDimensions,
    const int64_t *rhsContractingDimensions) {
  return wrap(mlir::stablehlo::DotDimensionNumbersAttr::get(
      unwrap(ctx),
      llvm::ArrayRef(lhsBatchingDimensions, nLhsBatchingDimensions),
      llvm::ArrayRef(rhsBatchingDimensions, nRhsBatchingDimensions),
      llvm::ArrayRef(lhsContractingDimensions, nLhsContractingDimensions),
      llvm::ArrayRef(rhsContractingDimensions, nRhsContractingDimensions)));
}

bool stablehloAttributeIsADotDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::DotDimensionNumbersAttr>();
}

intptr_t stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsBatchingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsBatchingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getLhsContractingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::DotDimensionNumbersAttr>()
      .getRhsContractingDimensions()[pos];
}

//===----------------------------------------------------------------------===//
// ConvDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloConvDimensionNumbersGet(
    MlirContext ctx, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    intptr_t nInputSpatialDimensions, const int64_t *inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    intptr_t nKernelSpatialDimensions, const int64_t *kernelSpatialDimensions,
    int64_t outputBatchDimension, int64_t outputFeatureDimension,
    intptr_t nOutputSpatialDimensions, const int64_t *outputSpatialDimensions) {
  return wrap(mlir::stablehlo::ConvDimensionNumbersAttr::get(
      unwrap(ctx), inputBatchDimension, inputFeatureDimension,
      llvm::ArrayRef(inputSpatialDimensions, nInputSpatialDimensions),
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      llvm::ArrayRef(kernelSpatialDimensions, nKernelSpatialDimensions),
      outputBatchDimension, outputFeatureDimension,
      llvm::ArrayRef(outputSpatialDimensions, nOutputSpatialDimensions)));
}

bool stablehloAttributeIsAConvDimensionNumbers(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ConvDimensionNumbersAttr>();
}

int64_t stablehloConvDimensionNumbersGetInputBatchDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputBatchDimension();
}

int64_t stablehloConvDimensionNumbersGetInputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getInputSpatialDimensions()[pos];
}

int64_t stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelInputFeatureDimension();
}

int64_t stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelOutputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getKernelSpatialDimensions()[pos];
}

int64_t stablehloConvDimensionNumbersGetOutputBatchDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputBatchDimension();
}

int64_t stablehloConvDimensionNumbersGetOutputFeatureDimension(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::ConvDimensionNumbersAttr>()
      .getOutputSpatialDimensions()[pos];
}

//===----------------------------------------------------------------------===//
// OutputOperandAlias
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloOutputOperandAliasGet(
    MlirContext ctx, intptr_t nOutputTupleIndices,
    const int64_t *outputTupleIndices, int64_t operandIndex,
    intptr_t nOperandTupleIndices, const int64_t *operandTupleIndices) {
  return wrap(mlir::stablehlo::OutputOperandAliasAttr::get(
      unwrap(ctx), llvm::ArrayRef(outputTupleIndices, nOutputTupleIndices),
      operandIndex, llvm::ArrayRef(operandTupleIndices, nOperandTupleIndices)));
}

bool stablehloAttributeIsAOutputOperandAlias(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::OutputOperandAliasAttr>();
}

intptr_t stablehloOutputOperandAliasGetOutputTupleIndicesSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::OutputOperandAliasAttr>()
      .getOutputTupleIndices()
      .size();
}

int64_t stablehloOutputOperandAliasGetOutputTupleIndicesElem(MlirAttribute attr,
                                                             intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::OutputOperandAliasAttr>()
      .getOutputTupleIndices()[pos];
}

int64_t stablehloOutputOperandAliasGetOperandIndex(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::OutputOperandAliasAttr>()
      .getOperandIndex();
}

intptr_t stablehloOutputOperandAliasGetOperandTupleIndicesSize(
    MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::OutputOperandAliasAttr>()
      .getOperandTupleIndices()
      .size();
}

int64_t stablehloOutputOperandAliasGetOperandTupleIndicesElem(
    MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::OutputOperandAliasAttr>()
      .getOperandTupleIndices()[pos];
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloComparisonDirectionAttrGet(MlirContext ctx,
                                                  MlirStringRef value) {
  std::optional<mlir::stablehlo::ComparisonDirection> comparisonDirection =
      mlir::stablehlo::symbolizeComparisonDirection(unwrap(value));
  if (!comparisonDirection) llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::stablehlo::ComparisonDirectionAttr::get(
      unwrap(ctx), comparisonDirection.value()));
}

bool stablehloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ComparisonDirectionAttr>();
}

MlirStringRef stablehloComparisonDirectionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonDirection(
      unwrap(attr)
          .cast<mlir::stablehlo::ComparisonDirectionAttr>()
          .getValue()));
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloComparisonTypeAttrGet(MlirContext ctx,
                                             MlirStringRef value) {
  std::optional<mlir::stablehlo::ComparisonType> comparisonType =
      mlir::stablehlo::symbolizeComparisonType(unwrap(value));
  if (!comparisonType) llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::stablehlo::ComparisonTypeAttr::get(unwrap(ctx),
                                                       comparisonType.value()));
}

bool stablehloAttributeIsAComparisonTypeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ComparisonTypeAttr>();
}

MlirStringRef stablehloComparisonTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonType(
      unwrap(attr).cast<mlir::stablehlo::ComparisonTypeAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// PrecisionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloPrecisionAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::stablehlo::Precision> precision =
      mlir::stablehlo::symbolizePrecision(unwrap(value));
  if (!precision) llvm::report_fatal_error("Invalid value.");
  return wrap(
      mlir::stablehlo::PrecisionAttr::get(unwrap(ctx), precision.value()));
}

bool stablehloAttributeIsAPrecisionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::PrecisionAttr>();
}

MlirStringRef stablehloPrecisionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyPrecision(
      unwrap(attr).cast<mlir::stablehlo::PrecisionAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// FftTypeAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloFftTypeAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::stablehlo::FftType> fftType =
      mlir::stablehlo::symbolizeFftType(unwrap(value));
  if (!fftType) llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::stablehlo::FftTypeAttr::get(unwrap(ctx), fftType.value()));
}

bool stablehloAttributeIsAFftTypeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::FftTypeAttr>();
}

MlirStringRef stablehloFftTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyFftType(
      unwrap(attr).cast<mlir::stablehlo::FftTypeAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// TransposeAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloTransposeAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::stablehlo::Transpose> transpose =
      mlir::stablehlo::symbolizeTranspose(unwrap(value));
  if (!transpose) llvm::report_fatal_error("Invalid value.");
  return wrap(
      mlir::stablehlo::TransposeAttr::get(unwrap(ctx), transpose.value()));
}

bool stablehloAttributeIsATransposeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::TransposeAttr>();
}

MlirStringRef stablehloTransposeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyTranspose(
      unwrap(attr).cast<mlir::stablehlo::TransposeAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// RngDistributionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloRngDistributionAttrGet(MlirContext ctx,
                                              MlirStringRef value) {
  std::optional<mlir::stablehlo::RngDistribution> rngDistribution =
      mlir::stablehlo::symbolizeRngDistribution(unwrap(value));
  if (!rngDistribution) llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::stablehlo::RngDistributionAttr::get(
      unwrap(ctx), rngDistribution.value()));
}

bool stablehloAttributeIsARngDistributionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::RngDistributionAttr>();
}

MlirStringRef stablehloRngDistributionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyRngDistribution(
      unwrap(attr).cast<mlir::stablehlo::RngDistributionAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloRngAlgorithmAttrGet(MlirContext ctx,
                                           MlirStringRef value) {
  std::optional<mlir::stablehlo::RngAlgorithm> rngAlgorithm =
      mlir::stablehlo::symbolizeRngAlgorithm(unwrap(value));
  if (!rngAlgorithm) llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::stablehlo::RngAlgorithmAttr::get(unwrap(ctx),
                                                     rngAlgorithm.value()));
}

bool stablehloAttributeIsARngAlgorithmAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::RngAlgorithmAttr>();
}

MlirStringRef stablehloRngAlgorithmAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyRngAlgorithm(
      unwrap(attr).cast<mlir::stablehlo::RngAlgorithmAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// ChannelHandle
//===----------------------------------------------------------------------===//

MlirAttribute stablehloChannelHandleGet(MlirContext ctx, int64_t handle,
                                        int64_t type) {
  return wrap(
      mlir::stablehlo::ChannelHandleAttr::get(unwrap(ctx), handle, type));
}

bool stablehloAttributeIsChannelHandle(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::ChannelHandleAttr>();
}

int64_t stablehloChannelHandleGetHandle(MlirAttribute attr) {
  return unwrap(attr).cast<mlir::stablehlo::ChannelHandleAttr>().getHandle();
}

int64_t stablehloChannelHandleGetType(MlirAttribute attr) {
  return unwrap(attr).cast<mlir::stablehlo::ChannelHandleAttr>().getType();
}

//===----------------------------------------------------------------------===//
// TypeExtensions
//===----------------------------------------------------------------------===//

MlirAttribute stablehloTypeExtensionsGet(MlirContext ctx, intptr_t nBounds,
                                         const int64_t *bounds) {
  return wrap(mlir::stablehlo::TypeExtensionsAttr::get(
      unwrap(ctx), llvm::ArrayRef(bounds, nBounds)));
}

bool stablehloAttributeIsTypeExtensions(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::stablehlo::TypeExtensionsAttr>();
}

intptr_t stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr) {
  return unwrap(attr)
      .cast<mlir::stablehlo::TypeExtensionsAttr>()
      .getBounds()
      .size();
}

int64_t stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos) {
  return unwrap(attr)
      .cast<mlir::stablehlo::TypeExtensionsAttr>()
      .getBounds()[pos];
}
