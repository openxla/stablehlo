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
  return unwrap(attr).isa<mlir::chlo::ComparisonDirectionAttr>();
}

MlirStringRef chloComparisonDirectionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::chlo::stringifyComparisonDirection(
      unwrap(attr).cast<mlir::chlo::ComparisonDirectionAttr>().getValue()));
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
  return unwrap(attr).isa<mlir::chlo::ComparisonTypeAttr>();
}

MlirStringRef chloComparisonTypeAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::chlo::stringifyComparisonType(
      unwrap(attr).cast<mlir::chlo::ComparisonTypeAttr>().getValue()));
}
