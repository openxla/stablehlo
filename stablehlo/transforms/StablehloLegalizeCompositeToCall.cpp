/* Copyright 2024 The StableHLO Authors.

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
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZECOMPOSITETOCALLPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

struct ReplaceCompositeWithCall final
    : OpRewritePattern<mlir::stablehlo::CompositeOp> {
  using OpRewritePattern::OpRewritePattern;

  ReplaceCompositeWithCall(MLIRContext *context)
      : OpRewritePattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult matchAndRewrite(CompositeOp op,
                                PatternRewriter &rewriter) const override {
    auto call = rewriter.create<mlir::func::CallOp>(
        op.getLoc(), op.getResultTypes(), op.getDecomposition(),
        op.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct StablehloLegalizeCompositeToCallPass
    : public impl::StablehloLegalizeCompositeToCallPassBase<
          StablehloLegalizeCompositeToCallPass> {
  using StablehloLegalizeCompositeToCallPassBase::
      StablehloLegalizeCompositeToCallPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    DenseSet<StringRef> excludedNames(exceptListOption.begin(),
                                      exceptListOption.end());

    ConversionTarget target(getContext());
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addDynamicallyLegalOp<stablehlo::CompositeOp>(
        [&](stablehlo::CompositeOp op) {
          return excludedNames.contains(op.getName());
        });

    RewritePatternSet patterns(context);
    patterns.add<ReplaceCompositeWithCall>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
};
}  // namespace

}  // namespace stablehlo
}  // namespace mlir
