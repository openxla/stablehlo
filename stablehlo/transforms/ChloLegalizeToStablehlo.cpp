/* Copyright 2020 The OpenXLA Authors.

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

// Enable the use of M_* math constants.
// NOTE: this must be first in the file to ensure that if cmath is transitively
// included by any other header it has the define set on first processing.
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants

#include "stablehlo/transforms/ChloLegalizeToStablehlo.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOSTABLEHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

struct ChloLegalizeToStablehloPass
    : public impl::ChloLegalizeToStablehloPassBase<
          ChloLegalizeToStablehloPass> {
  using ChloLegalizeToStablehloPassBase<
      ChloLegalizeToStablehloPass>::ChloLegalizeToStablehloPassBase;
  explicit ChloLegalizeToStablehloPass(bool legalizeBroadcasts,
                                       bool expandCompositions)
      : ChloLegalizeToStablehloPassBase<
            ChloLegalizeToStablehloPass>::ChloLegalizeToStablehloPassBase() {
    this->legalize_broadcasts_ = legalizeBroadcasts;
    this->expand_compositions_ = expandCompositions;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect, shape::ShapeDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ConversionTarget conversionTarget(getContext());
    RewritePatternSet conversionPatterns(&getContext());
    conversionTarget.addIllegalDialect<chlo::ChloDialect>();

    // Consider the stablehlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget
        .addLegalDialect<stablehlo::StablehloDialect, mlir::arith::ArithDialect,
                         mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                         mlir::shape::ShapeDialect, mlir::scf::SCFDialect>();
    conversionTarget.addLegalOp<chlo::MinimumBroadcastShapesOp>();

    if (legalize_broadcasts_) {
      populateChloBroadcastingPatterns(&getContext(), &conversionPatterns);
    }

    if (expand_compositions_) {
      populateDecomposeChloPatterns(&getContext(), &conversionPatterns);
    } else {
      conversionTarget
          .addLegalOp<chlo::NextAfterOp, chlo::PolygammaOp, chlo::ZetaOp>();
    }

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding stablehlo non-broadcasting op.
template <typename ChloOpTy, typename StablehloOpTy, typename Adaptor>
struct ConvertTrivialNonBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhsType =
        adaptor.getLhs().getType().template dyn_cast<RankedTensorType>();
    auto rhsType =
        adaptor.getRhs().getType().template dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType) return failure();

    // Requires rank broadcast.
    if (lhsType.getRank() != rhsType.getRank()) return failure();
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return failure();

    for (auto extents : llvm::zip(lhsType.getShape(), rhsType.getShape())) {
      auto lhsExtent = std::get<0>(extents);
      auto rhsExtent = std::get<1>(extents);
      if (lhsExtent != rhsExtent) {
        return failure();
      }
    }

    rewriter.replaceOp(op, Adaptor::createOp(op, op.getResult().getType(),
                                             adaptor.getOperands(), rewriter));
    return success();
  }
};

void populateChloBroadcastingPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns) {
  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  populateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
}

void populateDecomposeChloPatterns(MLIRContext *context,
                                   RewritePatternSet *patterns) {}

}  // namespace stablehlo
}  // namespace mlir
