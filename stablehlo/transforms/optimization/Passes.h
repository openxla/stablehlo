/* Copyright 2025 The StableHLO Authors.

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

#ifndef STABLEHLO_TRANSFORMS_OPTIMIZATION_PASSES_H
#define STABLEHLO_TRANSFORMS_OPTIMIZATION_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/optimization/Passes.h.inc"

/// Collection of canonicalization patterns for StableHLO.
void populateStablehloCanonicalizationPatterns(MLIRContext *context,
                                               RewritePatternSet *patterns,
                                               PatternBenefit benefit = 1);

/// Collection of folding patterns for StableHLO.
void populateStablehloAggressiveFolderPatterns(RewritePatternSet *patterns,
                                               MLIRContext *context,
                                               bool foldFloat,
                                               PatternBenefit benefit = 1);

/// A subset of folding patterns for StableHLO that is necessary for shape
/// refinement.
void populateStablehloShapeFolderPatterns(RewritePatternSet *patterns,
                                          MLIRContext *context,
                                          bool foldFloat = false,
                                          PatternBenefit benefit = 1);

/// Some workloads in XLA import StableHLO from HLO. Since there are a few
/// differences in HLO (no implicit captures, lots of tuples, etc.), this
/// set of patterns brings the imported HLO back to a more canonical form
/// without applying a full set of graph simplifications.
void populateStablehloHloImportCanonicalizationPatterns(
    MLIRContext *context, RewritePatternSet *patterns);
}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_OPTIMIZATION_PASSES_H
