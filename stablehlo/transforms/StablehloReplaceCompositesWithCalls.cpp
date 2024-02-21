// Copyright 2024 The StableHLO Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements composite inlining.

#include <cassert>
#include <functional>
#include <numeric>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOREPLACECOMPOSITESWITHCALLSPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

struct ReplaceCompositeWithCall final
    : OpRewritePattern<mlir::stablehlo::CompositeOp> {
  using OpRewritePattern::OpRewritePattern;

  ReplaceCompositeWithCall(MLIRContext *context,
                           const DenseSet<StringRef> &excludedNames_)
      : OpRewritePattern<mlir::stablehlo::CompositeOp>(context),
        excludedNames(excludedNames_) {}

  LogicalResult matchAndRewrite(CompositeOp op,
                                PatternRewriter &rewriter) const override {
    if (excludedNames.contains(op.getName()))
      return rewriter.notifyMatchFailure(
          op, Twine("excepted name: ") + op.getName());

    auto call = rewriter.create<mlir::func::CallOp>(
        op.getLoc(), op.getResultTypes(), op.getDecomposition(),
        op.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

 private:
  const DenseSet<StringRef> &excludedNames;
};

struct StablehloInlineCompositesPass
    : public impl::StablehloReplaceCompositesWithCallsPassBase<
          StablehloInlineCompositesPass> {
  using StablehloReplaceCompositesWithCallsPassBase::
      StablehloReplaceCompositesWithCallsPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto excludedNames =
        DenseSet<StringRef>(exceptListOption.begin(), exceptListOption.end());
    patterns.add<ReplaceCompositeWithCall>(context, excludedNames);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
};
}  // namespace

}  // namespace stablehlo
}  // namespace mlir
