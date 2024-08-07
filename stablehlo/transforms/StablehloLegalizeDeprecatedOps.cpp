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
#include <complex>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZEDEPRECATEDOPSPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

struct StablehloLegalizeDeprecatedOpsPass final
    : impl::StablehloLegalizeDeprecatedOpsPassBase<
          StablehloLegalizeDeprecatedOpsPass> {
  using StablehloLegalizeDeprecatedOpsPassBase::
      StablehloLegalizeDeprecatedOpsPassBase;

  LogicalResult initialize(MLIRContext *context) override {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalOp<BroadcastOp, CreateTokenOp, CrossReplicaSumOp, DotOp,
                         UnaryEinsumOp>();

    if (failOnUnusedOps) {
      // Deprecated ops to be removed with no replacements
      target->addIllegalOp<MapOp, RngOp>();
    }

    target->addLegalDialect<StablehloDialect>();

    RewritePatternSet patterns_(context);
    populateStablehloLegalizeDeprecatedOpsPatterns(context, &patterns_);
    patterns = std::move(patterns_);

    return success();
  }

  void runOnOperation() override {
    if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
      return signalPassFailure();
    }
  }

 private:
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};

///////////
// Patterns
///////////

struct CrossReplicaSumToAllReducePattern
    : public OpRewritePattern<CrossReplicaSumOp> {
  using OpRewritePattern<CrossReplicaSumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CrossReplicaSumOp crossReplicaSumOp,
                                PatternRewriter &rewriter) const override {
    auto allReduceOp = rewriter.replaceOpWithNewOp<AllReduceOp>(
        crossReplicaSumOp, crossReplicaSumOp.getType(),
        crossReplicaSumOp.getOperand(), crossReplicaSumOp.getReplicaGroups(),
        /*channel_handle=*/ChannelHandleAttr(),
        /*use_global_device_ids=*/false);

    auto *block = rewriter.createBlock(&allReduceOp.getComputation());
    auto elementType = RankedTensorType::get(
        {}, cast<ShapedType>(allReduceOp.getType(0)).getElementType());
    auto location = allReduceOp.getComputation().getLoc();
    block->addArguments({elementType, elementType}, {location, location});
    auto addOp = rewriter.create<AddOp>(location, block->getArgument(0),
                                        block->getArgument(1));
    rewriter.create<ReturnOp>(location, addOp.getResult());
    return success();
  }
};

///////////////////////
// DRR helper functions
///////////////////////

DenseElementsAttr getScalarOfType(Type ty, int64_t rawValue) {
  RankedTensorType scalarTy = RankedTensorType::get({}, ty);

  if (auto floatTy = mlir::dyn_cast<FloatType>(ty)) {
    APFloat value(floatTy.getFloatSemantics(), rawValue);
    return DenseElementsAttr::get(scalarTy, value);
  }
  if (auto intTy = mlir::dyn_cast<IntegerType>(ty)) {
    APInt value(intTy.getWidth(), static_cast<int64_t>(rawValue),
                /*isSigned=*/true);
    return DenseElementsAttr::get(scalarTy, value);
  }
  if (auto complexTy = mlir::dyn_cast<ComplexType>(ty)) {
    if (auto floatTy = mlir::cast<FloatType>(complexTy.getElementType())) {
      APFloat real(floatTy.getFloatSemantics(), rawValue);
      APFloat imag = APFloat::getZero(floatTy.getFloatSemantics());
      return DenseElementsAttr::get(scalarTy,
                                    std::complex<APFloat>(real, imag));
    }
  }
  llvm::report_fatal_error("unsupported type");
}

#include "stablehlo/transforms/StablehloLegalizeDeprecatedOpsPatterns.h.inc"
}  // namespace

void populateStablehloLegalizeDeprecatedOpsPatterns(
    MLIRContext *context, RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
  patterns->add<CrossReplicaSumToAllReducePattern>(context);
}

}  // namespace stablehlo
}  // namespace mlir
