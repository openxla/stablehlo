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

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloScatterDimensionNumbersGet(
    MlirContext ctx, intptr_t nUpdateWindowDims,
    const int64_t *updateWindowDims, intptr_t nInsertedWindowDims,
    const int64_t *insertedWindowDims, intptr_t nInputBatchingDims,
    const int64_t *inputBatchingDims, intptr_t nScatterIndicesBatchingDims,
    const int64_t *scatterIndicesBatchingDims,
    intptr_t nScatteredDimsToOperandDims,
    const int64_t *scatteredDimsToOperandDims, int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      unwrap(ctx), llvm::ArrayRef(updateWindowDims, nUpdateWindowDims),
      llvm::ArrayRef(insertedWindowDims, nInsertedWindowDims),
      llvm::ArrayRef(inputBatchingDims, nInputBatchingDims),
      llvm::ArrayRef(scatterIndicesBatchingDims, nScatterIndicesBatchingDims),
      llvm::ArrayRef(scatteredDimsToOperandDims, nScatteredDimsToOperandDims),
      indexVectorDim));
}

bool stablehloAttributeIsAScatterDimensionNumbers(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr));
}

intptr_t stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getUpdateWindowDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getUpdateWindowDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInsertedWindowDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInsertedWindowDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetInputBatchingDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInputBatchingDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetInputBatchingDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInputBatchingDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterIndicesBatchingDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterIndicesBatchingDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterDimsToOperandDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterDimsToOperandDims()[pos];
}

int64_t stablehloDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nOperandBatchingDims, const int64_t *operandBatchingDims,
    intptr_t nStartIndicesBatchingDims, const int64_t *startIndicesBatchingDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::GatherDimensionNumbersAttr::get(
      unwrap(ctx), llvm::ArrayRef(offsetDims, nOffsetDims),
      llvm::ArrayRef(collapsedSliceDims, nCollapsedSliceDims),
      llvm::ArrayRef(operandBatchingDims, nOperandBatchingDims),
      llvm::ArrayRef(startIndicesBatchingDims, nStartIndicesBatchingDims),
      llvm::ArrayRef(startIndexMap, nStartIndexMap), indexVectorDim));
}

bool stablehloAttributeIsAGatherDimensionNumbers(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr));
}

intptr_t stablehloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOffsetDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetOffsetDimsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOffsetDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getCollapsedSliceDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getCollapsedSliceDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOperandBatchingDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOperandBatchingDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndicesBatchingDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndicesBatchingDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetStartIndexMapSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndexMap()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetStartIndexMapElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndexMap()[pos];
}

int64_t stablehloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// DotAlgorithm
//===----------------------------------------------------------------------===//

MlirAttribute stablehloDotAlgorithmGet(
    MlirContext ctx, MlirType lhsPrecisionType, MlirType rhsPrecisionType,
    MlirType accumulationType, int64_t lhsComponentCount,
    int64_t rhsComponentCount, int64_t numPrimitiveOperations,
    bool allowImpreciseAccumulation) {
  return wrap(mlir::stablehlo::DotAlgorithmAttr::get(
      unwrap(ctx), unwrap(lhsPrecisionType), unwrap(rhsPrecisionType),
      unwrap(accumulationType), lhsComponentCount, rhsComponentCount,
      numPrimitiveOperations, allowImpreciseAccumulation));
}

bool stablehloAttributeIsADotAlgorithm(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr));
}

MlirType stablehloDotAlgorithmGetLhsPrecisionType(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr))
                  .getLhsPrecisionType());
}

MlirType stablehloDotAlgorithmGetRhsPrecisionType(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr))
                  .getRhsPrecisionType());
}

MlirType stablehloDotAlgorithmGetAccumulationType(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr))
                  .getAccumulationType());
}

int64_t stablehloDotAlgorithmGetLhsComponentCount(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr))
      .getLhsComponentCount();
}

int64_t stablehloDotAlgorithmGetRhsComponentCount(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr))
      .getRhsComponentCount();
}

int64_t stablehloDotAlgorithmGetNumPrimitiveOperations(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr))
      .getNumPrimitiveOperations();
}

bool stablehloDotAlgorithmGetAllowImpreciseAccumulation(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotAlgorithmAttr>(unwrap(attr))
      .getAllowImpreciseAccumulation();
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
  return llvm::isa<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr));
}

intptr_t stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsContractingDimensions()
      .size();
}

int64_t stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
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
  return llvm::isa<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr));
}

int64_t stablehloConvDimensionNumbersGetInputBatchDimension(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputBatchDimension();
}

int64_t stablehloConvDimensionNumbersGetInputFeatureDimension(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getInputSpatialDimensions()[pos];
}

int64_t stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelInputFeatureDimension();
}

int64_t stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelOutputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getKernelSpatialDimensions()[pos];
}

int64_t stablehloConvDimensionNumbersGetOutputBatchDimension(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getOutputBatchDimension();
}

int64_t stablehloConvDimensionNumbersGetOutputFeatureDimension(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getOutputFeatureDimension();
}

intptr_t stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
      .getOutputSpatialDimensions()
      .size();
}

int64_t stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ConvDimensionNumbersAttr>(unwrap(attr))
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
  return llvm::isa<mlir::stablehlo::OutputOperandAliasAttr>(unwrap(attr));
}

intptr_t stablehloOutputOperandAliasGetOutputTupleIndicesSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOutputTupleIndices()
      .size();
}

int64_t stablehloOutputOperandAliasGetOutputTupleIndicesElem(MlirAttribute attr,
                                                             intptr_t pos) {
  return llvm::cast<mlir::stablehlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOutputTupleIndices()[pos];
}

int64_t stablehloOutputOperandAliasGetOperandIndex(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOperandIndex();
}

intptr_t stablehloOutputOperandAliasGetOperandTupleIndicesSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::OutputOperandAliasAttr>(unwrap(attr))
      .getOperandTupleIndices()
      .size();
}

int64_t stablehloOutputOperandAliasGetOperandTupleIndicesElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::OutputOperandAliasAttr>(unwrap(attr))
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
  return llvm::isa<mlir::stablehlo::ComparisonDirectionAttr>(unwrap(attr));
}

MlirStringRef stablehloComparisonDirectionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonDirection(
      llvm::cast<mlir::stablehlo::ComparisonDirectionAttr>(unwrap(attr))
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
  return llvm::isa<mlir::stablehlo::ComparisonTypeAttr>(unwrap(attr));
}

MlirStringRef stablehloComparisonTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonType(
      llvm::cast<mlir::stablehlo::ComparisonTypeAttr>(unwrap(attr))
          .getValue()));
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
  return llvm::isa<mlir::stablehlo::PrecisionAttr>(unwrap(attr));
}

MlirStringRef stablehloPrecisionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyPrecision(
      llvm::cast<mlir::stablehlo::PrecisionAttr>(unwrap(attr)).getValue()));
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
  return llvm::isa<mlir::stablehlo::FftTypeAttr>(unwrap(attr));
}

MlirStringRef stablehloFftTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyFftType(
      llvm::cast<mlir::stablehlo::FftTypeAttr>(unwrap(attr)).getValue()));
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
  return llvm::isa<mlir::stablehlo::TransposeAttr>(unwrap(attr));
}

MlirStringRef stablehloTransposeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyTranspose(
      llvm::cast<mlir::stablehlo::TransposeAttr>(unwrap(attr)).getValue()));
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
  return llvm::isa<mlir::stablehlo::RngDistributionAttr>(unwrap(attr));
}

MlirStringRef stablehloRngDistributionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyRngDistribution(
      llvm::cast<mlir::stablehlo::RngDistributionAttr>(unwrap(attr))
          .getValue()));
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
  return llvm::isa<mlir::stablehlo::RngAlgorithmAttr>(unwrap(attr));
}

MlirStringRef stablehloRngAlgorithmAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyRngAlgorithm(
      llvm::cast<mlir::stablehlo::RngAlgorithmAttr>(unwrap(attr)).getValue()));
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
  return llvm::isa<mlir::stablehlo::ChannelHandleAttr>(unwrap(attr));
}

int64_t stablehloChannelHandleGetHandle(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ChannelHandleAttr>(unwrap(attr))
      .getHandle();
}

int64_t stablehloChannelHandleGetType(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ChannelHandleAttr>(unwrap(attr)).getType();
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
  return llvm::isa<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr));
}

intptr_t stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()
      .size();
}

int64_t stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()[pos];
}
