// Copyright 2024 The StableHLO Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements composite inlining.

#include <cassert>

#include "mlir/IR/PatternMatch.h"
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
