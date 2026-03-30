/* Copyright 2026 The StableHLO Authors. */

#include <utility>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

// --- 1. Helper Functions (Must be in mlir::stablehlo for TableGen) ---

Value getConstantLikeMaxFiniteValue(OpBuilder &b, Location loc, Value val) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(b, loc, llvm::APFloat::getLargest(ty.getFloatSemantics()), val);
}

Value getConstantLikeInfValue(OpBuilder &b, Location loc, Value val, bool negative) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(b, loc, llvm::APFloat::getInf(ty.getFloatSemantics(), negative), val);
}

// --- 2. Forward Declarations for Population ---

void populateStablehloComplexMathExpanderPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns);
void populateStablehloComplexFullMathExpanderPatterns(MLIRContext *context,
                                                      RewritePatternSet *patterns);

#define GEN_PASS_DEF_STABLEHLOCOMPLEXMATHEXPANDERPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

// --- 3. Include Patterns (Generated code uses helpers above) ---

#include "stablehlo/transforms/StablehloComplexMathExpanderPatterns.h.inc"

// --- 4. Pass Implementation ---

struct StablehloComplexMathExpanderPass
    : public impl::StablehloComplexMathExpanderPassBase<StablehloComplexMathExpanderPass> {
  
  // Use the Base class constructor to properly initialize 'enableFullExpansion'
  using StablehloComplexMathExpanderPassBase::StablehloComplexMathExpanderPassBase;

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Default accuracy patterns
    populateStablehloComplexMathExpanderPatterns(context, &patterns);

    // Full expansion (Mul/Div) logic
    if (enableFullExpansion) {
      populateStablehloComplexFullMathExpanderPatterns(context, &patterns);
    }

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      func.emitError("Failed to converge StableHLOComplexMathExpanderPass");
      signalPassFailure();
    }
  }
};

} // namespace

// --- 5. Definition of Population Functions ---

void populateStablehloComplexMathExpanderPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns) {
  // Logic to populate standard patterns
  populateWithGenerated(*patterns);
}

void populateStablehloComplexFullMathExpanderPatterns(MLIRContext *context,
                                                      RewritePatternSet *patterns) {
  // If you split the .td files later, this would call populateFullWithGenerated.
  // For now, it shares the same generated pool.
  populateWithGenerated(*patterns);
}

} // namespace stablehlo
} // namespace mlir