/* Copyright 2026 The StableHLO Authors. */

#include <utility>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

Value getConstantLikeMaxFiniteValue(OpBuilder &b, Location loc, Value val) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(b, loc, llvm::APFloat::getLargest(ty.getFloatSemantics()), val);
}

Value getConstantLikeInfValue(OpBuilder &b, Location loc, Value val, bool negative) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(b, loc, llvm::APFloat::getInf(ty.getFloatSemantics(), negative), val);
}

#define GEN_PASS_DEF_STABLEHLOCOMPLEXMATHEXPANDERPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

namespace default_patterns {
#include "stablehlo/transforms/StablehloComplexMathExpanderPatterns.h.inc"
} // namespace default_patterns

namespace full_patterns {
#include "stablehlo/transforms/StablehloComplexFullMathExpanderPatterns.h.inc"
} // namespace full_patterns


struct StablehloComplexMathExpanderPass
    : public impl::StablehloComplexMathExpanderPassBase<StablehloComplexMathExpanderPass> {
  
  using StablehloComplexMathExpanderPassBase::StablehloComplexMathExpanderPassBase;

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    default_patterns::populateWithGenerated(patterns);
    if (enableFullExpansion) {
      full_patterns::populateWithGenerated(patterns);
    }

    GreedyRewriteConfig config;
    
    if (failed(applyPatternsGreedily(func, std::move(patterns), config))) {
      func.emitError("Failed to converge StableHLOComplexMathExpanderPass");
      signalPassFailure();
    }
  }
};

} // namespace


void populateStablehloComplexMathExpanderPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns) {
  default_patterns::populateWithGenerated(*patterns);
}

void populateStablehloComplexFullMathExpanderPatterns(MLIRContext *context,
                                                      RewritePatternSet *patterns) {
  full_patterns::populateWithGenerated(*patterns);
}

} // namespace stablehlo
} // namespace mlir