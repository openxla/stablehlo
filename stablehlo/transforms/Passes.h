/* Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_TRANSFORMS_PASSES_H
#define STABLEHLO_TRANSFORMS_PASSES_H

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/Quant/IR/Quant.h"   // IWYU pragma: keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DECL

std::unique_ptr<::mlir::Pass> createStablehloAggressiveSimplificationPass(
    GreedyRewriteConfig config);

#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/Passes.h.inc"

// Populates --stablehlo-canonicalize-dynamism patterns.
void populateStablehloCanonicalizeDynamismPatterns(RewritePatternSet *patterns,
                                                   MLIRContext *context);

// Populates --stablehlo-refine-shapes patterns.
void populateStablehloRefineShapesPatterns(RewritePatternSet *patterns,
                                           MLIRContext *context);

// Populates StableHLO ops to VHLO ops rewriting patterns.
void populateStablehloToVhloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates VHLO ops to StableHLO ops rewriting patterns.
void populateVhloToStablehloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates VHLO downgrade rewriting patterns.
void populateVhloToVersionPatterns(RewritePatternSet *patterns,
                                   TypeConverter *converter,
                                   MLIRContext *contexts);

/// Collection of rewrite patterns for lowering of CHLO ops to StableHLO and
/// Shape ops.
void populateChloToStablehloPatterns(MLIRContext *context,
                                     RewritePatternSet *patterns);

/// CHLO ConstantLikeOp to StableHLO ConstantOp
/// May require dynamic shape broadcasting.
void populateChloConstantLikePattern(MLIRContext *context,
                                     RewritePatternSet *patterns);

/// Collection of rewrite patterns for lowering quantized StableHLO operations
/// using uniform dequantize/quantize operations.
void populateStablehloLegalizeQuantizedOpToQDQPatterns(
    RewritePatternSet *patterns, MLIRContext *context,
    PatternBenefit benefit = 1);

/// Collection of rewrite patterns for composing quantized StableHLO operations
/// using unform dequantize/quantize operations.
void populateStablehloLegalizeQDQToQuantizedOpPatterns(
    RewritePatternSet *patterns, MLIRContext *context);

/// Collection of patterns to upgrade deprecated ops to long-term supported ops.
void populateStablehloLegalizeDeprecatedOpsPatterns(
    MLIRContext *context, RewritePatternSet *patterns);

/// Collection of shape dialect to StableHLO patterns.
void populateShapeToStablehloPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns);

/// Collection of patterns to create compatibility expander for StableHLO
/// operations.
void populateStablehloCompatibilityExpanderPatterns(
    RewritePatternSet *patterns, MLIRContext *context,
    vhlo::Version targetVersion);

//// Additional pass constructors ////

std::unique_ptr<OperationPass<ModuleOp>> createStablehloRefineArgumentsPass(
    TypeRange refinedTypes);

/// Creates a pass that wraps StableHLO ops in CompositeOp.
/// The pass takes in a map from op's type id to a function that returns the
/// attributes to be added to the CompositeOp. The pass also takes in a
/// version number for the CompositeOp.
using CompositeAttributeProvider =
    std::function<std::optional<NamedAttrList>(Operation *)>;
using CompositeAttributeProviderMap =
    llvm::DenseMap<mlir::TypeID, CompositeAttributeProvider>;
std::unique_ptr<OperationPass<ModuleOp>> createStablehloWrapInCompositePass(
    const CompositeAttributeProviderMap &compositeAttributeProviderMap,
    int32_t compositeVersion);

/// Wraps the given operation in a CompositeOp with the specified NamedAttrs and
/// version and returns the CompositeOp.
///
/// **A typical usage **
///
/// ```cpp
/// // To wrap a specific stablehlo.add instance
///
/// mlir::stablehlo::AddOp addOp = ...; // The op instanced to be wrapped.
/// mlir::ModuleOp module = addOp->getParentOfType<mlir::ModuleOp>();
/// mlir::OpBuilder builder(addOp);
/// mlir::NamedAttrList attrs = ...; // Attributes to be set on the
///                                  // composite op.
/// int32_t version = 0; // Composite version.
///
/// mlir::stablehlo::CompositeOp compositeOp =
///   mlir::stablehlo::wrapOperationInComposite(builder, addOp, attrs,
///                                             version, module);
/// addOp.replaceAllUsesWith(compositeOp);
/// ```
stablehlo::CompositeOp wrapOperationInComposite(OpBuilder &builder,
                                                Operation *op,
                                                const NamedAttrList &attrs,
                                                int32_t compositeVersion,
                                                ModuleOp module);
//// Pass pipelines ////

// StableHLO consumers can add this pipeline to convert portable artifacts to
// StableHLO programs. This pipeline will silently pass if programs are not
// portable artifacts.
//
// Uses vhlo-to-version and vhlo-legalize-to-stablehlo passes. Does not require
// an option to specify VHLO target version since it always converts VHLO to
// the current version in order to legalize to StableHLO.
void createStablehloDeserializePipeline(OpPassManager &pm);

// Creates a pipeline of StableHLO-specific MLIR passes to remove dynamism from
// the program. This is achieved via refining the "main" function's arguments
// and propagating new shapes throughout the program argument types and shapes
// within an MLIR module. The main function is either a function with name
// "main", if there are multiple functions, or the single function within the
// module.
//
// This pipeline focuses on:
//   1. Refining function argument types based on provided `refinedTypes`.
//   2. Refining shape information of operations within functions.
//   3. Replaces dynamic StableHLO ops with the corresponding static
//   counterparts if applicable.
void createStablehloRemoveDynamismPipeline(OpPassManager &pm,
                                           TypeRange refinedTypes);

// Decomposes quantized operations within a StableHLO module by
// applying a series of MLIR passes essentially breaking down the quantized
// operations into a primitive math operations.
void createStablehloLowerQuantPipeline(OpPassManager &pm);

/// Collection of patterns to create expander for StableHLO complex
/// math operations.
void populateStablehloComplexMathExpanderPatterns(RewritePatternSet *patterns,
                                                  MLIRContext *context);

// Adds `stablehlo-deserialize` pipeline as a registered pass pipeline
// for opt tools.
void registerPassPipelines();

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_PASSES_H
