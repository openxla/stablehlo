/* Copyright 2024 The StableHLO Authors. All Rights Reserved.
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

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
#define GEN_PASS_DEF_STABLEHLOCOMPLEXMATHEXPANDERPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

static Value getConstantLikeMaxFiniteValue(OpBuilder &b, Location loc,
                                           Value val) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getLargest(ty.getFloatSemantics()), val);
}

static Value getConstantLikeInfValue(OpBuilder &b, Location loc, Value val,
                                     bool negative) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getInf(ty.getFloatSemantics(), negative), val);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct StablehloComplexMathExpanderPass
    : public impl::StablehloComplexMathExpanderPassBase<
          StablehloComplexMathExpanderPass> {
  StablehloComplexMathExpanderPass()
      : StablehloComplexMathExpanderPassBase<
            StablehloComplexMathExpanderPass>() {}

 public:
  LogicalResult initialize(MLIRContext *context) override {
    config.setUseTopDownTraversal(true);
    RewritePatternSet patterns_(context);
    populateStablehloComplexMathExpanderPatterns(context, &patterns_);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsGreedily(func, patterns, config))) {
      func.emitError("Failed to converge StableHLOComplexMathExpanderPass in ")
          << config.getMaxIterations() << " iterations";
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

#include "stablehlo/transforms/StablehloComplexMathExpanderPatterns.h.inc"

}  // namespace

void populateStablehloComplexMathExpanderPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
}

}  // namespace stablehlo
}  // namespace mlir
