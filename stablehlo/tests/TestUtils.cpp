/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#include "stablehlo/tests/TestUtils.h"

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace hlo {

namespace {

struct InferReturnTypesPattern : public RewritePattern {
  explicit InferReturnTypesPattern(MLIRContext *context)
      : RewritePattern("hlo_test_infer.get_return_types", 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) return failure();
    auto *definingOp = op->getOperand(0).getDefiningOp();
    auto definingOpInt =
        llvm::dyn_cast_or_null<InferTypeOpInterface>(definingOp);
    if (!definingOpInt) return failure();
    SmallVector<Type, 4> types;
    if (failed(definingOpInt.inferReturnTypes(
            op->getContext(), op->getLoc(), definingOp->getOperands(),
            definingOp->getAttrDictionary(), definingOp->getPropertiesStorage(),
            definingOp->getRegions(), types)))
      return failure();

    // Replace the op with another pass-through op with attributes added.
    OperationState state(op->getLoc(), "hlo_test_infer.return_types",
                         op->getOperands(), op->getResultTypes(),
                         op->getAttrs());
    auto *newOp = rewriter.create(state);
    for (const auto &it : llvm::enumerate(types))
      newOp->setAttr((StringRef("types") + Twine(it.index())).str(),
                     TypeAttr::get(it.value()));
    rewriter.replaceOp(op, {newOp->getResults()});
    return success();
  }
};

struct ReifyReturnTypeShapesPattern : public RewritePattern {
  explicit ReifyReturnTypeShapesPattern(MLIRContext *context)
      : RewritePattern("hlo_test_infer.reify_return_type_shapes", 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) return failure();
    auto definingOp =
        op->getOperand(0).getDefiningOp<InferShapedTypeOpInterface>();
    if (!definingOp) return failure();
    SmallVector<Value, 4> returnShapes;
    if (failed(definingOp.reifyReturnTypeShapes(
            rewriter, definingOp->getOperands(), returnShapes)))
      return failure();
    rewriter.replaceOp(op, returnShapes);
    return success();
  }
};

LogicalResult checkSpeculatability(PatternRewriter &rewriter, Operation *op,
                                   mlir::Speculation::Speculatability spec) {
  if (op->getNumOperands() != 1) return failure();
  auto definingOp =
      op->getOperand(0).getDefiningOp<ConditionallySpeculatable>();
  if (!definingOp || !definingOp->hasOneUse()) return failure();

  if (definingOp.getSpeculatability() == spec) {
    rewriter.eraseOp(op);
    rewriter.eraseOp(definingOp);
    return success();
  }

  return failure();
}

struct IsSpeculatablePattern : public RewritePattern {
  explicit IsSpeculatablePattern(MLIRContext *context)
      : RewritePattern("hlo_test_speculatability.is_speculatable", 1, context) {
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return checkSpeculatability(rewriter, op, mlir::Speculation::Speculatable);
  }
};

struct IsRecursivelySpeculatablePattern : public RewritePattern {
  explicit IsRecursivelySpeculatablePattern(MLIRContext *context)
      : RewritePattern("hlo_test_speculatability.is_recursively_speculatable",
                       1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return checkSpeculatability(rewriter, op,
                                mlir::Speculation::RecursivelySpeculatable);
  }
};

struct IsNotSpeculatablePattern : public RewritePattern {
  explicit IsNotSpeculatablePattern(MLIRContext *context)
      : RewritePattern("hlo_test_speculatability.is_not_speculatable", 1,
                       context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return checkSpeculatability(rewriter, op,
                                mlir::Speculation::NotSpeculatable);
  }
};

#define GEN_PASS_DEF_HLOTESTINFERPASS
#define GEN_PASS_DEF_HLOTESTSPECULATABILITYPASS
#include "stablehlo/tests/TestUtils.h.inc"

struct HloTestInferPass : public impl::HloTestInferPassBase<HloTestInferPass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet patterns_(context);
    patterns_.add<InferReturnTypesPattern>(context);
    patterns_.add<ReifyReturnTypeShapesPattern>(context);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

 private:
  FrozenRewritePatternSet patterns;
};

struct HloTestSpeculatabilityPass
    : public impl::HloTestSpeculatabilityPassBase<HloTestSpeculatabilityPass> {
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet patterns_(context);
    patterns_.add<IsSpeculatablePattern>(context);
    patterns_.add<IsNotSpeculatablePattern>(context);
    patterns_.add<IsRecursivelySpeculatablePattern>(context);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.maxIterations = 1;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  FrozenRewritePatternSet patterns;
};

#define GEN_PASS_REGISTRATION
#include "stablehlo/tests/TestUtils.h.inc"

}  // namespace

void registerAllTestPasses() { registerHloTestPasses(); }

}  // namespace hlo
}  // namespace mlir
