/* Copyright 2022 OpenXLA Authors. All Rights Reserved.

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

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "stablehlo-prepare-for-tosa"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {

#define GEN_PASS_DEF_STABLEHLOPREPAREFORTOSAPASS
#include "stablehlo/conversions/tosa/transforms/Passes.h.inc"

namespace {

class StablehloPrepareForTosaPass
    : public impl::StablehloPrepareForTosaPassBase<
          StablehloPrepareForTosaPass> {
 public:
  explicit StablehloPrepareForTosaPass() = default;
  void runOnOperation() override;
};

void StablehloPrepareForTosaPass::runOnOperation() {
  auto* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  // Currently these equivalents are not available here.
  // TODO: Enable post upstreaming decision.
  // stablehlo::DotGeneralOp::getCanonicalizationPatterns(patterns, ctx);
  // stablehlo::populateGeneralDotOpLoweringPatterns(&patterns, ctx);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace

}  // namespace tosa
}  // namespace mlir
