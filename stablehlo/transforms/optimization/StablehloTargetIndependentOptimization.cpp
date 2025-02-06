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

#include <cassert>
#include <cstdint>
#include <utility>

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOTARGETINDEPENDENTOPTIMIZATIONPASS
#include "stablehlo/transforms/optimization/Passes.h.inc"

// This is an upper limit on how many elements can be folded by an op folder.
// This limit doesn't apply to some special cases like adding a zero,
// multiplying by one, doing many operations with splats.
constexpr int64_t kFoldOpEltLimit = 65536;

struct StablehloTargetIndependentOptimizationPass
    : public impl::StablehloTargetIndependentOptimizationPassBase<
          StablehloTargetIndependentOptimizationPass> {
  using StablehloTargetIndependentOptimizationPassBase::
      StablehloTargetIndependentOptimizationPassBase;

  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    bool foldFloat = false;
    populateStablehloCanonicalizationPatterns(context, &patterns_);
    populateStablehloAggressiveFolderPatterns(&patterns_, context, foldFloat,
                                              /*benefit=*/2);
    patterns = std::move(patterns_);

    return success();
  }

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.fold = true;
    config.cseConstants = true;
    config.maxIterations = kFoldOpEltLimit;
    config.useTopDownTraversal = false;
    if (failed(applyPatternsGreedily(getOperation(), patterns, config)))
      signalPassFailure();
  }

 private:
  FrozenRewritePatternSet patterns;
};

}  // namespace stablehlo
}  // namespace mlir
