/* Copyright 2022 The StableHLO Authors.
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

#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/MapStablehloToVhlo.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/TypeConversion.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace vhlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOVHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

#define RETURN_CONVERTED_ENUM_ATTR(Name)                             \
  auto stablehloValue = stablehlo::stringify##Name(attr.getValue()); \
  auto hloValue = vhlo::symbolize##Name(stablehloValue);             \
  if (!hloValue.has_value()) return {};                              \
  return vhlo::Name##Attr::get(attr.getContext(), hloValue.value())

Attribute convertAttrToVhlo(Attribute stablehloAttr) {
  // Handle StableHLO attributes.
  // The logic that handles attributes from other dialects (e.g. builtin
  // attributes) lives below.
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::ChannelHandleAttr>()) {
    return vhlo::ChannelHandleAttr::get(attr.getContext(), attr.getHandle(),
                                        attr.getType());
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::ComparisonDirectionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::ComparisonTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::ConvDimensionNumbersAttr>()) {
    return vhlo::ConvDimensionNumbersAttr::get(
        attr.getContext(), attr.getInputBatchDimension(),
        attr.getInputFeatureDimension(), attr.getInputSpatialDimensions(),
        attr.getKernelInputFeatureDimension(),
        attr.getKernelOutputFeatureDimension(),
        attr.getKernelSpatialDimensions(), attr.getOutputBatchDimension(),
        attr.getOutputFeatureDimension(), attr.getOutputSpatialDimensions());
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::CustomCallApiVersionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::DotDimensionNumbersAttr>()) {
    return vhlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::FftTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::GatherDimensionNumbersAttr>()) {
    return vhlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::PrecisionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Precision);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::RngAlgorithmAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm);
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::RngDistributionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::ScatterDimensionNumbersAttr>()) {
    return vhlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getScatterDimsToOperandDims(),
        attr.getIndexVectorDim());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::TransposeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose);
  }
  if (stablehloAttr.getDialect().getNamespace() ==
      stablehlo::StablehloDialect::getDialectNamespace()) {
    // Our guiding principle is to support all StableHLO functionality in
    // vhlo. This check is here only for exceptional situations, e.g.
    // when we added a new StableHLO attribute and forgot to update the code
    // above.
    return {};
  }

  // Handle non-StableHLO attributes.
  // If an attribute is not defined in StableHLO, then it is unchanged,
  // with the exception of ArrayAttr which is converted recursively.
  if (auto stablehloAttrs = stablehloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> hloAttrs;
    for (auto stablehloAttr : stablehloAttrs) {
      auto hloAttr = convertAttrToVhlo(stablehloAttr);
      if (!hloAttr) return {};
      hloAttrs.push_back(hloAttr);
    }
    return ArrayAttr::get(stablehloAttrs.getContext(), hloAttrs);
  }
  return stablehloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

template <typename StablehloOpTy>
class StablehloToVhloOpConverter : public OpConversionPattern<StablehloOpTy> {
 public:
  using OpConversionPattern<StablehloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      StablehloOpTy stablehloOp, typename StablehloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SmallVector<Type> versionedTypes;
    if (failed(this->getTypeConverter()->convertTypes(
            stablehloOp->getResultTypes(), versionedTypes))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed type conversion\n");
      return failure();
    }

    // These operands have already been converted to StableHLO by
    // the dialect conversion infrastructure.
    ValueRange stablehloOperands = adaptor.getOperands();

    SmallVector<NamedAttribute> stablehloAttrs;
    for (NamedAttribute hloAttr : stablehloOp->getAttrs()) {
      auto stablehloAttr = convertAttrToVhlo(hloAttr.getValue());
      if (!stablehloAttr) return failure();
      stablehloAttrs.push_back({hloAttr.getName(), stablehloAttr});
    }

    // Convert the vhlo operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // vhlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    StablehloToVhloOp<StablehloOpTy> versionedOp;
    if constexpr (std::is_same<StablehloOpTy, stablehlo::CaseOp>::value) {
      versionedOp = rewriter.replaceOpWithNewOp<vhlo::CaseOpV1>(
          stablehloOp, versionedTypes, stablehloOperands, stablehloAttrs,
          stablehloOp.getBranches().size());
    } else {
      versionedOp =
          rewriter.replaceOpWithNewOp<StablehloToVhloOp<StablehloOpTy>>(
              stablehloOp, versionedTypes, stablehloOperands, stablehloAttrs);
    }

    for (auto [hloRegion, stablehloRegion] :
         llvm::zip(stablehloOp->getRegions(), versionedOp->getRegions())) {
      rewriter.inlineRegionBefore(hloRegion, stablehloRegion,
                                  stablehloRegion.end());
    }
    return success();
  }
};

template <typename... StablehloOpTypes>
void populateStablehloToVhloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  patterns->add<StablehloToVhloOpConverter<StablehloOpTypes>...>(*converter,
                                                                 context);
}

}  // namespace

//////////////////////////
/// StableHLO --> VHLO ///
//////////////////////////

struct StablehloLegalizeToVhloPass
    : public impl::StablehloLegalizeToVhloPassBase<
          StablehloLegalizeToVhloPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<vhlo::VhloDialect>();

    vhlo::StablehloToVhloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    vhlo::populateStablehloToVhloPatterns(&patterns, &converter, &getContext());
    registerFuncOpsForTypeConversion(target, patterns, converter);

    // StableHLO is a subset of VHLO.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed partial conversion\n");
      return signalPassFailure();
    }
  }
};

void populateStablehloToVhloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  populateStablehloToVhloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}
}  // namespace vhlo
}  // namespace mlir