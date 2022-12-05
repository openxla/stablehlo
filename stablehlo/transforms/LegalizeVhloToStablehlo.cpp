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
namespace stablehlo {

#define GEN_PASS_DEF_VHLOLEGALIZETOSTABLEHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

//////////////////////////
/// VHLO --> StableHLO ///
//////////////////////////
#define RETURN_CONVERTED_ENUM_ATTR(Name)                       \
  auto vhloValue = vhlo::stringify##Name(attr.getValue());     \
  auto stablehloValue = stablehlo::symbolize##Name(vhloValue); \
  if (!stablehloValue.has_value()) return {};                  \
  return stablehlo::Name##Attr::get(attr.getContext(), stablehloValue.value())

Attribute convertAttrToStablehlo(Attribute vhloAttr) {
  LLVM_DEBUG(llvm::dbgs() << "Converting " << vhloAttr);
  if (auto attr = vhloAttr.dyn_cast<vhlo::ChannelHandleAttr>()) {
    return stablehlo::ChannelHandleAttr::get(attr.getContext(),
                                             attr.getHandle(), attr.getType());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ComparisonDirectionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ComparisonTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ConvDimensionNumbersAttr>()) {
    return stablehlo::ConvDimensionNumbersAttr::get(
        attr.getContext(), attr.getInputBatchDimension(),
        attr.getInputFeatureDimension(), attr.getInputSpatialDimensions(),
        attr.getKernelInputFeatureDimension(),
        attr.getKernelOutputFeatureDimension(),
        attr.getKernelSpatialDimensions(), attr.getOutputBatchDimension(),
        attr.getOutputFeatureDimension(), attr.getOutputSpatialDimensions());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::CustomCallApiVersionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::DotDimensionNumbersAttr>()) {
    return stablehlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::FftTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::GatherDimensionNumbersAttr>()) {
    return stablehlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::OutputOperandAliasAttr>()) {
    return stablehlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::PrecisionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Precision);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::RngAlgorithmAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::RngDistributionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ScatterDimensionNumbersAttr>()) {
    return stablehlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getScatterDimsToOperandDims(),
        attr.getIndexVectorDim());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::TransposeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose);
  }
  if (vhloAttr.getDialect().getNamespace() ==
      vhlo::VhloDialect::getDialectNamespace()) {
    // All VHLO attributes must have counterparts in StableHLO.
    return {};
  }

  // Handle non-VHLO attributes.
  // If an attribute is not defined in vhlo, then it is unchanged,
  // with the exception of ArrayAttr which is converted recursively.
  // This will change once we fork necessary upstream types to VHLO.
  if (auto vhloAttrs = vhloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto vhloAttr : vhloAttrs) {
      auto stablehloAttr = convertAttrToStablehlo(vhloAttr);
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back(stablehloAttr);
    }
    return ArrayAttr::get(vhloAttrs.getContext(), stablehloAttrs);
  }
  return vhloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

struct VhloLegalizeToStablehloPass
    : public impl::VhloLegalizeToStablehloPassBase<
          VhloLegalizeToStablehloPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<vhlo::VhloDialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();

    vhlo::VhloToStablehloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateVhloToStablehloPatterns(&patterns, &converter,
                                               &getContext());
    registerFuncOpsForTypeConversion(target, patterns, converter);

    // VHLO should always be convertible to StableHLO if upgraded.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

template <typename VhloOpTy>
class VhloToStablehloOpConverter : public OpConversionPattern<VhloOpTy> {
 public:
  using OpConversionPattern<VhloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      VhloOpTy vhloOp, typename VhloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SmallVector<Type> stablehloTypes;
    if (failed(this->getTypeConverter()->convertTypes(vhloOp->getResultTypes(),
                                                      stablehloTypes)))
      return failure();

    // These operands have already been converted to StableHLO by
    // the dialect conversion infrastructure.
    ValueRange stablehloOperands = adaptor.getOperands();

    SmallVector<NamedAttribute> stablehloAttrs;
    for (NamedAttribute vhloAttr : vhloOp->getAttrs()) {
      auto stablehloAttr = convertAttrToStablehlo(vhloAttr.getValue());
      if (!stablehloAttr) return failure();
      stablehloAttrs.push_back({vhloAttr.getName(), stablehloAttr});
    }

    // Convert the vhlo operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // vhlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    VhloToStablehloOp<VhloOpTy> stablehloOp;
    if constexpr (std::is_same<VhloOpTy, vhlo::CaseOpV1>::value) {
      stablehloOp = rewriter.replaceOpWithNewOp<stablehlo::CaseOp>(
          vhloOp, stablehloTypes, stablehloOperands, stablehloAttrs,
          vhloOp.getBranches().size());
    } else {
      stablehloOp = rewriter.replaceOpWithNewOp<VhloToStablehloOp<VhloOpTy>>(
          vhloOp, stablehloTypes, stablehloOperands, stablehloAttrs);
    }

    for (auto [vhloRegion, stablehloRegion] :
         llvm::zip(vhloOp->getRegions(), stablehloOp->getRegions())) {
      rewriter.inlineRegionBefore(vhloRegion, stablehloRegion,
                                  stablehloRegion.end());
    }
    return success();
  }
};

template <typename... StablehloOpTypes>
void populateVhloToStablehloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  patterns
      ->add<VhloToStablehloOpConverter<StablehloToVhloOp<StablehloOpTypes>>...>(
          *converter, context);
}

}  // namespace

void populateVhloToStablehloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  populateVhloToStablehloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

}  // namespace stablehlo
}  // namespace mlir
