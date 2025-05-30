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
#include <memory>
#include <utility>

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/transforms/optimization/Passes.h"

namespace mlir::stablehlo {

#define GEN_PASS_DEF_STABLEHLOTARGETINDEPENDENTOPTIMIZATIONPASS
#include "stablehlo/transforms/optimization/Passes.h.inc"

struct StablehloTargetIndependentOptimizationPass
    : public impl::StablehloTargetIndependentOptimizationPassBase<
          StablehloTargetIndependentOptimizationPass> {
  explicit StablehloTargetIndependentOptimizationPass(
      StablehloTargetIndependentOptimizationPassOptions options,
      GreedyRewriteConfig rewriteConfig = {})
      : StablehloTargetIndependentOptimizationPassBase(options),
        rewriteConfig(rewriteConfig) {}

  explicit StablehloTargetIndependentOptimizationPass()
      : StablehloTargetIndependentOptimizationPassBase() {}

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    StablehloAggressiveFolderPassOptions folderOptions{
        /*assumeNoUndeclaredSideEffects=*/assumeNoUndeclaredSideEffects,
        /*foldOpElementLimit=*/foldOpElementLimit,
        /*optimizeFloat=*/optimizeFloat,
    };
    StablehloAggressiveSimplificationPassOptions simplificationOptions{
        /*foldOpElementLimit=*/foldOpElementLimit,
    };

    populateStablehloAggressiveFolderPatterns(context, &patterns, folderOptions,
                                              /*benefit=*/2);
    populateStablehloCanonicalizationPatterns(context, &patterns,
                                              simplificationOptions);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     rewriteConfig)))
      signalPassFailure();
  }

 private:
  GreedyRewriteConfig rewriteConfig;
};

std::unique_ptr<::mlir::Pass> createStablehloTargetIndependentOptimizationPass(
    StablehloTargetIndependentOptimizationPassOptions options,
    GreedyRewriteConfig rewriteConfig) {
  return std::make_unique<StablehloTargetIndependentOptimizationPass>(
      options, rewriteConfig);
}

}  // namespace mlir::stablehlo
