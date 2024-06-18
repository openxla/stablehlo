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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"  // Include for TypeConverter
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZEQUANTIZEDOPTOQDQPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

bool isAnyQuantizedTypes(TypeRange types) {
  return llvm::any_of(types, [](Type type) {
    return isa<quant::QuantizedType>(getElementTypeOrSelf(type));
  });
}

template <typename StablehloOpType>
struct QuantizedStablehloOpConversion
    : public OpRewritePattern<StablehloOpType> {
  using OpRewritePattern<StablehloOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(StablehloOpType op,
                                PatternRewriter& rewriter) const override {
    if (!isAnyQuantizedTypes(op->getOperandTypes()) &&
        !isAnyQuantizedTypes(op->getResultTypes())) {
      return failure();
    }

    SmallVector<Value> dequantizedOperands;
    for (auto operand : op->getOperands()) {
      if (isa<quant::QuantizedType>(getElementTypeOrSelf(operand.getType()))) {
        dequantizedOperands.push_back(
            rewriter.create<UniformDequantizeOp>(op->getLoc(), operand));
      } else {
        dequantizedOperands.push_back(operand);
      }
    }

    auto origOp = op.getOperation();
    auto origAttrs = origOp->getAttrs();
    auto newOp = rewriter
                     .create<StablehloOpType>(op.getLoc(), dequantizedOperands,
                                              origAttrs)
                     .getOperation();

    SmallVector<Value> quantizedResults;
    for (auto [oldResult, newResult] :
         llvm::zip(origOp->getResults(), newOp->getResults())) {
      if (isa<quant::QuantizedType>(
              getElementTypeOrSelf(oldResult.getType()))) {
        quantizedResults.push_back(
            rewriter.create<stablehlo::UniformQuantizeOp>(
                op->getLoc(), oldResult.getType(), newResult));
      } else {
        quantizedResults.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, quantizedResults);
    return success();
  }
};

class StablehloLegalizeQuantizedOpToQDQPass
    : public impl::StablehloLegalizeQuantizedOpToQDQPassBase<
          StablehloLegalizeQuantizedOpToQDQPass> {
 public:
  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    populateStablehloLegalizeQuantizedOpToQDQPatterns(&patterns_, context);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, patterns, config))) {
      func.emitError("Failed to converge StablehloLegalizeQuantizedOpToQDQ in ")
          << config.maxIterations << " iterations";
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

template <typename... StablehloOpTypes>
void populateStablehloLegalizeQuantizedOpToQDQPatterns(
    RewritePatternSet* patterns, MLIRContext* context) {
  patterns->add<QuantizedStablehloOpConversion<StablehloOpTypes>...>(context);
}

}  // namespace

void populateStablehloLegalizeQuantizedOpToQDQPatterns(
    RewritePatternSet* patterns, MLIRContext* context) {
  // The following list covers most of the operations which, according to the
  // stablehlo spoecification document, interprets the quantized
  // operation using dequant-op-quant strategy. The ones excluded are
  // AddOP, ConvolutionOp, DotGeneralOp, and DynamicConvOp, which are current
  // using `stablehlo-legalize-quant-to-int` pass for decomposituion to
  // primitive math operations.
  populateStablehloLegalizeQuantizedOpToQDQPatterns<
      AbsOp, Atan2Op, BatchNormGradOp, BatchNormInferenceOp,
      BatchNormTrainingOp, CbrtOp, CeilOp, CholeskyOp, ClampOp, CompareOp,
      CosineOp, DivOp, Expm1Op, ExpOp, FloorOp, Log1pOp, LogisticOp, LogOp,
      MaxOp, MinOp, MulOp, NegOp, PowOp, ReducePrecisionOp, RemOp, RoundOp,
      RoundNearestEvenOp, RsqrtOp, SelectOp, SignOp, SineOp, SqrtOp, SubtractOp,
      TanhOp, TriangularSolveOp>(patterns, context);
}

}  // namespace stablehlo
}  // namespace mlir
