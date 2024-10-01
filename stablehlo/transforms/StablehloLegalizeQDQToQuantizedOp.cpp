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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"  // Include for TypeConverter
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZEQDQTOQUANTIZEDOPPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

struct QuantizedStablehloQDQToQuantizedOpConversion
    : public OpRewritePattern<stablehlo::UniformQuantizeOp> {
  using OpRewritePattern<stablehlo::UniformQuantizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::UniformQuantizeOp quantOp,
                                PatternRewriter& rewriter) const override {
    // Matching sequence of ops:
    //    UniformDequantizeOp -> non-quantized Op -> UniformQuantizeOp.
    // Start matching from a UniformQuantizeOp (`quantOp`).
    // Get the Op (`computeOp`) which defines the inputs of the `quantOp`.
    // Verify all inputs of the `computeOp` are produced by
    // UniformDequantizeOp (`dequantOp`).
    // Note: The pass does not delete any prexisting op.
    auto* computeOp = quantOp->getOperand(0).getDefiningOp();
    if (!computeOp)
      return rewriter.notifyMatchFailure(
          quantOp, "requires operand to be defined by an op");

    if (computeOp->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(computeOp,
                                         "ops with regions are not supported");

    if (computeOp->getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          computeOp, "ops with variadic results are not supported");

    if (isAnyQuantizedTypes(computeOp->getOperandTypes()))
      return rewriter.notifyMatchFailure(computeOp,
                                         "requires non quantized operands");

    // Collect quantized operands and result types to rewrite.
    // All operands and results must be quantized
    llvm::SmallVector<Value> quantizedComputeOpOperands;
    for (const Value& operand : computeOp->getOperands()) {
      auto* definingOp = operand.getDefiningOp();
      if (!definingOp)
        return rewriter.notifyMatchFailure(
            computeOp, "requires operand to be defined by an op");

      auto dequantOp = dyn_cast<stablehlo::UniformDequantizeOp>(definingOp);
      if (!dequantOp)
        return rewriter.notifyMatchFailure(
            definingOp,
            "requires operand to be defined by an stablehlo.uniform_dequantize "
            "op");

      quantizedComputeOpOperands.push_back(dequantOp->getOperand(0));
    }

    rewriter.setInsertionPointAfter(computeOp);
    OperationState newState(computeOp->getLoc(),
                            computeOp->getName().getStringRef(),
                            quantizedComputeOpOperands,
                            quantOp->getResultTypes(), computeOp->getAttrs());
    Operation* quantizedComputeOp = rewriter.create(newState);

    // Now that `computeOp` is quantized, replace all uses of the `quantOp`
    // with the `quantizedComputeOp`'s result.
    quantOp.getResult().replaceAllUsesWith(quantizedComputeOp->getResult(0));

    return success();
  }
};

class StablehloLegalizeQDQToQuantizedOpPass
    : public impl::StablehloLegalizeQDQToQuantizedOpPassBase<
          StablehloLegalizeQDQToQuantizedOpPass> {
 public:
  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    populateStablehloLegalizeQDQToQuantizedOpPatterns(&patterns_, context);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, patterns, config))) {
      func.emitError(
          "Failed to converge StablehloLegalizeQDQToQuantizedOpPass in ")
          << config.maxIterations << " iterations";
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

}  // namespace

void populateStablehloLegalizeQDQToQuantizedOpPatterns(
    RewritePatternSet* patterns, MLIRContext* context) {
  patterns->add<QuantizedStablehloQDQToQuantizedOpConversion>(context);
}

}  // namespace stablehlo
}  // namespace mlir
