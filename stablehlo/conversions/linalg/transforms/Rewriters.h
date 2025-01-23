/* Copyright 2022 The IREE Authors
   Copyright 2023 OpenXLA Authors. All Rights Reserved.

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

#ifndef STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H
#define STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::stablehlo {

//===----------------------------------------------------------------------===//
// General StableHLO lowering patterns.
//===----------------------------------------------------------------------===//

/// Populates the patterns that convert from StableHLO to Linalg on tensors.
void populateStablehloToLinalgConversionPatterns(MLIRContext *context,
                                                 TypeConverter &typeConverter,
                                                 RewritePatternSet *patterns,
                                                 bool enablePrimitiveOps,
                                                 bool enableSparseOps);

//===----------------------------------------------------------------------===//
// Fine-grained patterns used by the implementation.
//===----------------------------------------------------------------------===//
namespace detail {
/// Populates the patterns that convert from elementwise StableHLO ops to Linalg
/// on tensors.
void populatePointwiseStablehloToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns, bool enablePrimitiveOps);

/// Populates the patterns that convert from convolution StableHLO ops to Linalg
/// on tensors.
void populateStablehloConvolutionToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

/// Populates the patterns that convert from dot product StableHLO ops to Linalg
/// on tensors.
void populateStablehloDotProdToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

/// Populates the patterns that convert from random number generation StableHLO
/// ops to Linalg on tensors.
void populateStablehloRandomToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

/// Populates the patterns that convert from reduction StableHLO ops to Linalg
/// on tensors.
void populateStablehloReductionToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns, bool enablePrimitiveOps);

/// Populates the patterns that convert scalar StableHLO ops to Arith ops.
void populateScalarHloToArithConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns,
    llvm::function_ref<bool(Operation *)> filterFn = nullptr);
}  // namespace detail

}  // namespace mlir::stablehlo

#endif  // STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H
