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

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace vhlo {
#define GEN_PASS_DECL_STABLEHLOLEGALIZETOVHLOPASS
#define GEN_PASS_DECL_VHLOLEGALIZETOSTABLEHLOPASS
#define GEN_PASS_DECL_VHLOUPGRADEPASS
#define GEN_PASS_DECL_VHLODOWNGRADEPASS
#define GEN_PASS_DECL_VHLOTOVERSIONPASS
#include "stablehlo/transforms/CompatibilityPasses.h.inc"

/// Registers all transformation passes.
void registerStablehloCompatibilityPasses();

// Populates StableHLO ops to Versioned StableHLO ops rewriting patterns.
void populateStablehloToVhloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates Versioned StableHLO ops to StableHLO ops rewriting patterns.
void populateVhloToStablehloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates Versioned StableHLO downgrade rewriting patterns.
void populateVhloToVersionPatterns(RewritePatternSet *patterns,
                                   TypeConverter *converter,
                                   MLIRContext *contexts);
}  // namespace vhlo
}  // namespace mlir