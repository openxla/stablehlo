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
#ifndef STABLEHLO_INTEGRATIONS_C_CHLO_ATTRIBUTES_H
#define STABLEHLO_INTEGRATIONS_C_CHLO_ATTRIBUTES_H

#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
chloComparisonDirectionAttrGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED bool chloAttributeIsAComparisonDirectionAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
chloComparisonDirectionAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute chloComparisonTypeAttrGet(MlirContext ctx,
                                                           MlirStringRef value);

MLIR_CAPI_EXPORTED bool chloAttributeIsAComparisonTypeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
chloComparisonTypeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// RaggedDotDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute chloRaggedDotDimensionNumbersGet(
    MlirContext ctx,                                                        //
    intptr_t nLhsBatchingDimensions, const int64_t *lhsBatchingDimensions,  //
    intptr_t nRhsBatchingDimensions, const int64_t *rhsBatchingDimensions,  //
    intptr_t nLhsContractingDimensions,                                     //
    const int64_t *lhsContractingDimensions,                                //
    intptr_t nRhsContractingDimensions,                                     //
    const int64_t *rhsContractingDimensions,                                //
    intptr_t nLhsRaggedDimensions,                                          //
    const int64_t *lhsRaggedDimensions,                                     //
    intptr_t nRhsGroupDimensions,                                           //
    const int64_t *rhsGroupDimensions);

MLIR_CAPI_EXPORTED bool chloAttributeIsARaggedDotDimensionNumbers(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
chloRaggedDotDimensionNumbersGetLhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
chloRaggedDotDimensionNumbersGetLhsBatchingDimensionsElem(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
chloRaggedDotDimensionNumbersGetRhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
chloRaggedDotDimensionNumbersGetRhsBatchingDimensionsElem(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
chloRaggedDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
chloRaggedDotDimensionNumbersGetLhsContractingDimensionsElem(MlirAttribute attr,
                                                             intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
chloRaggedDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
chloRaggedDotDimensionNumbersGetRhsContractingDimensionsElem(MlirAttribute attr,
                                                             intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
chloRaggedDotDimensionNumbersGetLhsRaggedDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
chloRaggedDotDimensionNumbersGetLhsRaggedDimensionsElem(MlirAttribute attr,
                                                        intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
chloRaggedDotDimensionNumbersGetRhsGroupDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
chloRaggedDotDimensionNumbersGetRhsGroupDimensionsElem(MlirAttribute attr,
                                                       intptr_t pos);

//===----------------------------------------------------------------------===//
// PrecisionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute chloPrecisionAttrGet(MlirContext ctx,
                                                      MlirStringRef value);

MLIR_CAPI_EXPORTED bool chloAttributeIsAPrecisionAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef chloPrecisionAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // STABLEHLO_INTEGRATIONS_C_CHLO_ATTRIBUTES_H
