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
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"  // Include for TypeConverter
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZEATOBPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

// move this to common quant_util file
bool isAnyQuantizedTypes(TypeRange types) {
  return llvm::any_of(types, [](Type type) {
    return isa<quant::QuantizedType>(getElementTypeOrSelf(type));
  });
}

struct QuantizedStablehloAToBOpConversion
    : public OpRewritePattern<stablehlo::UniformQuantizeOp> {
  using OpRewritePattern<stablehlo::UniformQuantizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::UniformQuantizeOp op,
                                PatternRewriter& rewriter) const override {
    auto* quantizingOp = op->getOperand(0).getDefiningOp();
    // Ignore UniformQuantized op withput the parent op
    if (!(quantizingOp)) return failure();

    // if quantizingOp is alredy quanitzed, no work to do
    if (isAnyQuantizedTypes(quantizingOp->getOperandTypes())) return failure();

    llvm::SmallVector<Value> quantizedOperands;
    for (const auto& operandx : quantizingOp->getOperands()) {
      // backtrack and get the quantized operand
      // quantizedOp -> DequantizedOp -> shloOp
      auto* opx = operandx.getDefiningOp();
      // It is possible an operand is input of the module.
      if (!(opx)) {
        quantizedOperands.push_back(operandx);
        continue;
      }

      if (auto deqOp = dyn_cast<stablehlo::UniformDequantizeOp>(opx)) {
        quantizedOperands.push_back(deqOp->getOperand(0));
      } else {
        // can argument is result of another quantized op and not
        // UniformDequantizedOp? ignore this use case for now.
        return failure();
      }
    }

    SmallVector<Type> quantizedResults;
    for (auto newResult : op->getResultTypes()) {
      quantizedResults.push_back(newResult);
    }
    rewriter.setInsertionPointAfter(quantizingOp);
    OperationState newState(
        quantizingOp->getLoc(), quantizingOp->getName().getStringRef(),
        quantizedOperands, quantizedResults, quantizingOp->getAttrs());
    for (unsigned int i = 0; i < quantizingOp->getNumRegions(); ++i) {
      newState.addRegion();
    }
    Operation* quantizedOp = rewriter.create(newState);

    if (quantizingOp->getNumRegions() != 0) {
      for (const auto& QuantizingOpRegion :
           llvm::enumerate(quantizingOp->getRegions())) {
        Region& newRegion =
            quantizedOp->getRegion(QuantizingOpRegion.index());
        mlir::IRMapping mapping;
        QuantizingOpRegion.value().cloneInto(&newRegion, mapping);
      }
    }
    op.getResult().replaceAllUsesWith(quantizedOp->getResult(0));

    return success();
  }
};

class StablehloLegalizeAToBPass
    : public impl::StablehloLegalizeAToBPassBase<StablehloLegalizeAToBPass> {
 public:
  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    populateStablehloLegalizeAToBPatterns(&patterns_, context);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, patterns, config))) {
      func.emitError("Failed to converge StablehloLegalizeAToB in ")
          << config.maxIterations << " iterations";
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

}  // namespace

void populateStablehloLegalizeAToBPatterns(RewritePatternSet* patterns,
                                           MLIRContext* context,
                                           PatternBenefit benefit) {
  patterns->add<QuantizedStablehloAToBOpConversion>(context);
}

}  // namespace stablehlo
}  // namespace mlir
