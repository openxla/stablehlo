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

#include "stablehlo/integrations/c/ChloAttributes.h"

#include <optional>

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "stablehlo/dialect/ChloOps.h"

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MlirAttribute chloComparisonDirectionAttrGet(MlirContext ctx,
                                             MlirStringRef value) {
  std::optional<mlir::chlo::ComparisonDirection> comparisonDirection =
      mlir::chlo::symbolizeComparisonDirection(unwrap(value));
  if (!comparisonDirection) llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::chlo::ComparisonDirectionAttr::get(
      unwrap(ctx), comparisonDirection.value()));
}

bool chloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return llvm::isa<mlir::chlo::ComparisonDirectionAttr>(unwrap(attr));
}

MlirStringRef chloComparisonDirectionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::chlo::stringifyComparisonDirection(
      llvm::cast<mlir::chlo::ComparisonDirectionAttr>(unwrap(attr))
          .getValue()));
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MlirAttribute chloComparisonTypeAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::chlo::ComparisonType> comparisonType =
      mlir::chlo::symbolizeComparisonType(unwrap(value));
  if (!comparisonType) llvm::report_fatal_error("Invalid value.");
  return wrap(
      mlir::chlo::ComparisonTypeAttr::get(unwrap(ctx), comparisonType.value()));
}

bool chloAttributeIsAComparisonTypeAttr(MlirAttribute attr) {
  return llvm::isa<mlir::chlo::ComparisonTypeAttr>(unwrap(attr));
}

MlirStringRef chloComparisonTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::chlo::stringifyComparisonType(
      llvm::cast<mlir::chlo::ComparisonTypeAttr>(unwrap(attr)).getValue()));
}

//===----------------------------------------------------------------------===//
// RaggedDotDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute chloRaggedDotDimensionNumbersGet(
    MlirContext ctx, intptr_t nLhsBatchingDimensions,
    const int64_t *lhsBatchingDimensions, intptr_t nRhsBatchingDimensions,
    const int64_t *rhsBatchingDimensions, intptr_t nLhsContractingDimensions,
    const int64_t *lhsContractingDimensions, intptr_t nRhsContractingDimensions,
    const int64_t *rhsContractingDimensions, intptr_t nLhsRaggedDimensions,
    const int64_t *lhsRaggedDimensions, intptr_t nRhsGroupDimensions,
    const int64_t *rhsGroupDimensions) {
  return wrap(mlir::chlo::RaggedDotDimensionNumbersAttr::get(
      unwrap(ctx),
      llvm::ArrayRef(lhsBatchingDimensions, nLhsBatchingDimensions),
      llvm::ArrayRef(rhsBatchingDimensions, nRhsBatchingDimensions),
      llvm::ArrayRef(lhsContractingDimensions, nLhsContractingDimensions),
      llvm::ArrayRef(rhsContractingDimensions, nRhsContractingDimensions),
      llvm::ArrayRef(lhsRaggedDimensions, nLhsRaggedDimensions),
      llvm::ArrayRef(rhsGroupDimensions, nRhsGroupDimensions)));
}

bool chloAttributeIsARaggedDotDimensionNumbers(MlirAttribute attr) {
  return llvm::isa<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr));
}

intptr_t chloRaggedDotDimensionNumbersGetLhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()
      .size();
}

int64_t chloRaggedDotDimensionNumbersGetLhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()[pos];
}

intptr_t chloRaggedDotDimensionNumbersGetRhsBatchingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()
      .size();
}

int64_t chloRaggedDotDimensionNumbersGetRhsBatchingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()[pos];
}

intptr_t chloRaggedDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()
      .size();
}

int64_t chloRaggedDotDimensionNumbersGetLhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()[pos];
}

intptr_t chloRaggedDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getRhsContractingDimensions()
      .size();
}

int64_t chloRaggedDotDimensionNumbersGetRhsContractingDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getRhsContractingDimensions()[pos];
}

intptr_t chloRaggedDotDimensionNumbersGetLhsRaggedDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getLhsRaggedDimensions()
      .size();
}

int64_t chloRaggedDotDimensionNumbersGetLhsRaggedDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getLhsRaggedDimensions()[pos];
}

intptr_t chloRaggedDotDimensionNumbersGetRhsGroupDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getRhsGroupDimensions()
      .size();
}

int64_t chloRaggedDotDimensionNumbersGetRhsGroupDimensionsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::chlo::RaggedDotDimensionNumbersAttr>(unwrap(attr))
      .getRhsGroupDimensions()[pos];
}

//===----------------------------------------------------------------------===//
// PrecisionAttr
//===----------------------------------------------------------------------===//

MlirAttribute chloPrecisionAttrGet(MlirContext ctx, MlirStringRef value) {
  std::optional<mlir::chlo::Precision> precision =
      mlir::chlo::symbolizePrecision(unwrap(value));
  if (!precision) llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::chlo::PrecisionAttr::get(unwrap(ctx), precision.value()));
}

bool chloAttributeIsAPrecisionAttr(MlirAttribute attr) {
  return llvm::isa<mlir::chlo::PrecisionAttr>(unwrap(attr));
}

MlirStringRef chloPrecisionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::chlo::stringifyPrecision(
      llvm::cast<mlir::chlo::PrecisionAttr>(unwrap(attr)).getValue()));
}
