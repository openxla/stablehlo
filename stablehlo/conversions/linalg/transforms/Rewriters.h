// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H
#define STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_REWRITERS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::stablehlo {

//===----------------------------------------------------------------------===//
// General StableHLO/CHLO lowering patterns.
//===----------------------------------------------------------------------===//

/// Populates the patterns that convert from StableHLO to Linalg on tensors.
void populateStableHloToLinalgConversionPatterns(MLIRContext *context,
                                                 TypeConverter &typeConverter,
                                                 RewritePatternSet *patterns,
                                                 bool enablePrimitiveOps);

/// Collection of rewrite patterns for lowering of CHLO ops to StableHLO and
/// Shape ops.
void populateLegalizeChloPatterns(MLIRContext *context,
                                  RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering of StableHLO ops to SCF control
/// flow ops.
void populateLegalizeControlFlowPatterns(MLIRContext *context,
                                         RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering of StableHLO dim operations.
void populateLegalizeShapeComputationPatterns(MLIRContext *context,
                                              RewritePatternSet *patterns);

//===----------------------------------------------------------------------===//
// Fine-grained patterns used by the implementation.
//===----------------------------------------------------------------------===//
namespace detail {
/// Populates the patterns that convert from elementwise StableHLO ops to Linalg
/// on tensors.
void populatePointwiseStableHloToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns, bool enablePrimitiveOps);

/// Populates the patterns that convert from convolution StableHLO ops to Linalg
/// on tensors.
void populateStableHloConvolutionToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

/// Populates the patterns that convert from dot product StableHLO ops to Linalg
/// on tensors.
void populateStableHloDotProdToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

/// Populates the patterns that convert from random number generation StableHLO
/// ops to Linalg on tensors.
void populateStableHloRandomToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns);

/// Populates the patterns that convert from reduction StableHLO ops to Linalg
/// on tensors.
void populateStableHloReductionToLinalgConversionPatterns(
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
