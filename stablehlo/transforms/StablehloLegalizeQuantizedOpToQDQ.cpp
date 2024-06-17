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

#define DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(OpName)                            \
  struct Quantized##OpName##Conversion : public OpRewritePattern<OpName> {     \
    using OpRewritePattern<OpName>::OpRewritePattern;                          \
    LogicalResult matchAndRewrite(OpName op,                                   \
                                  PatternRewriter& rewriter) const override {  \
      if (!isAnyQuantizedTypes(op->getOperandTypes()) &&                       \
          !isAnyQuantizedTypes(op->getResultTypes())) {                        \
        return failure();                                                      \
      }                                                                        \
                                                                               \
      SmallVector<Value> dequantizedOperands;                                  \
      for (auto operand : op->getOperands()) {                                 \
        if (isa<quant::QuantizedType>(                                         \
                getElementTypeOrSelf(operand.getType()))) {                    \
          dequantizedOperands.push_back(                                       \
              rewriter.create<UniformDequantizeOp>(op->getLoc(), operand));    \
        } else {                                                               \
          dequantizedOperands.push_back(operand);                              \
        }                                                                      \
      }                                                                        \
                                                                               \
      auto origOp = op.getOperation();                                         \
      auto origAttrs = origOp->getAttrs();                                     \
      auto newOp =                                                             \
          rewriter.create<OpName>(op.getLoc(), dequantizedOperands, origAttrs) \
              .getOperation();                                                 \
                                                                               \
      SmallVector<Value> quantizedResults;                                     \
      for (auto [oldResult, newResult] :                                       \
           llvm::zip(origOp->getResults(), newOp->getResults())) {             \
        if (isa<quant::QuantizedType>(                                         \
                getElementTypeOrSelf(oldResult.getType()))) {                  \
          quantizedResults.push_back(                                          \
              rewriter.create<stablehlo::UniformQuantizeOp>(                   \
                  op->getLoc(), oldResult.getType(), newResult));              \
        } else {                                                               \
          quantizedResults.push_back(newResult);                               \
        }                                                                      \
      }                                                                        \
      rewriter.replaceOp(op, quantizedResults);                                \
      return success();                                                        \
    }                                                                          \
  };

// The following list covers most of the operations which, according to the
// stablehlo spoecification document, interprets the quantized
// operation using dequant-op-quant strategy. The ones excluded are
// AddOP, ConvolutionOp, DotGeneralOp, and DynamicConvOp, which are current
// using `stablehlo-legalize-quant-to-int` pass for decomposituion to primitive
// math operations.
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(AbsOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(Atan2Op)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(BatchNormGradOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(BatchNormInferenceOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(BatchNormTrainingOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(CbrtOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(CeilOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(CholeskyOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(ClampOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(CompareOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(CosineOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(DivOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(Expm1Op)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(ExpOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(FloorOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(Log1pOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(LogisticOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(LogOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(MaxOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(MinOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(MulOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(NegOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(PowOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(ReducePrecisionOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(RemOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(RoundOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(RoundNearestEvenOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(RsqrtOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(SelectOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(SignOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(SineOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(SqrtOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(SubtractOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(TanhOp)
DEFINE_QUANT_REWRITE_TO_QDQ_PATTERN(TriangularSolveOp)

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
      func.emitError("Failed to converge StablehloCanonicalizeDynamism in ")
          << config.maxIterations << " iterations";
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

}  // namespace

void populateStablehloLegalizeQuantizedOpToQDQPatterns(
    RewritePatternSet* patterns, MLIRContext* context) {
  patterns->add<QuantizedAbsOpConversion>(context);
  patterns->add<QuantizedAtan2OpConversion>(context);
  patterns->add<QuantizedBatchNormGradOpConversion>(context);
  patterns->add<QuantizedBatchNormInferenceOpConversion>(context);
  patterns->add<QuantizedBatchNormTrainingOpConversion>(context);
  patterns->add<QuantizedCbrtOpConversion>(context);
  patterns->add<QuantizedCeilOpConversion>(context);
  patterns->add<QuantizedCholeskyOpConversion>(context);
  patterns->add<QuantizedClampOpConversion>(context);
  patterns->add<QuantizedCompareOpConversion>(context);
  patterns->add<QuantizedCosineOpConversion>(context);
  patterns->add<QuantizedDivOpConversion>(context);
  patterns->add<QuantizedExpm1OpConversion>(context);
  patterns->add<QuantizedExpOpConversion>(context);
  patterns->add<QuantizedFloorOpConversion>(context);
  patterns->add<QuantizedLog1pOpConversion>(context);
  patterns->add<QuantizedLogisticOpConversion>(context);
  patterns->add<QuantizedLogOpConversion>(context);
  patterns->add<QuantizedMaxOpConversion>(context);
  patterns->add<QuantizedMinOpConversion>(context);
  patterns->add<QuantizedMulOpConversion>(context);
  patterns->add<QuantizedNegOpConversion>(context);
  patterns->add<QuantizedPowOpConversion>(context);
  patterns->add<QuantizedReducePrecisionOpConversion>(context);
  patterns->add<QuantizedRemOpConversion>(context);
  patterns->add<QuantizedRoundOpConversion>(context);
  patterns->add<QuantizedRoundNearestEvenOpConversion>(context);
  patterns->add<QuantizedRsqrtOpConversion>(context);
  patterns->add<QuantizedSelectOpConversion>(context);
  patterns->add<QuantizedSignOpConversion>(context);
  patterns->add<QuantizedSineOpConversion>(context);
  patterns->add<QuantizedSqrtOpConversion>(context);
  patterns->add<QuantizedSubtractOpConversion>(context);
  patterns->add<QuantizedTanhOpConversion>(context);
  patterns->add<QuantizedTriangularSolveOpConversion>(context);
}

}  // namespace stablehlo
}  // namespace mlir
