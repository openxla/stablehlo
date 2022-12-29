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

#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/MapStablehloToVhlo.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/TypeConversion.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOVHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

#define RETURN_CONVERTED_ENUM_ATTR(Name)                             \
  auto stablehloValue = stablehlo::stringify##Name(attr.getValue()); \
  auto vhloValue = vhlo::symbolize##Name(stablehloValue);            \
  if (!vhloValue.has_value()) return {};                             \
  return vhlo::Name##Attr::get(attr.getContext(), vhloValue.value())

Attribute convertCustomCallApiVersion(Attribute stablehloAttr) {
  if (auto attr = stablehloAttr.dyn_cast<CustomCallApiVersionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion);
  }
  return {};
}

Attribute convertAttrToVhlo(Attribute stablehloAttr,
                            TypeConverter* typeConverter) {
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
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::OutputOperandAliasAttr>()) {
    return vhlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
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
    // All StableHLO attributes must have counterparts in VHLO.
    return {};
  }

  // Handle builtin attributes.
  // Supported attributes are wrapped or converted to VHLO.

  if (auto floatAttr = stablehloAttr.dyn_cast<FloatAttr>()) {
    auto floatType = typeConverter->convertType(floatAttr.getType());
    if (!floatType) return {};
    return vhlo::FloatV1Attr::get(floatAttr.getContext(), floatType,
                                  floatAttr.getValue());
  }

  if (auto flatSymAttr = stablehloAttr.dyn_cast<FlatSymbolRefAttr>()) {
    auto rootRef =
        convertAttrToVhlo(flatSymAttr.getRootReference(), typeConverter);
    if (!rootRef) return {};
    return vhlo::FlatSymbolRefV1Attr::get(flatSymAttr.getContext(), rootRef);
  }

  if (auto strAttr = stablehloAttr.dyn_cast<StringAttr>()) {
    if (!strAttr.getType().isa<NoneType>()) {
      // Don't support custom string types
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to convert string with type: " << strAttr << '\n');
      return {};
    }
    return vhlo::StringV1Attr::get(strAttr.getContext(), strAttr.getValue());
  }

  if (auto unitAttr = stablehloAttr.dyn_cast<UnitAttr>()) {
    return vhlo::UnitV1Attr::get(unitAttr.getContext());
  }

  if (auto intAttr = stablehloAttr.dyn_cast<IntegerAttr>()) {
    return vhlo::IntegerV1Attr::get(intAttr.getContext(), intAttr);
  }

  if (auto stablehloAttrs = stablehloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> vhloAttrs;
    for (auto stablehloAttr : stablehloAttrs) {
      auto vhloAttr = convertAttrToVhlo(stablehloAttr, typeConverter);
      if (!vhloAttr) return {};
      vhloAttrs.push_back(vhloAttr);
    }
    return vhlo::ArrayV1Attr::get(stablehloAttrs.getContext(), vhloAttrs);
  }
  if (auto elementsAttr = stablehloAttr.dyn_cast<DenseIntOrFPElementsAttr>()) {
    auto vhloType = typeConverter->convertType(elementsAttr.getType());
    if (!vhloType) return {};
    return vhlo::DenseIntOrFPElementsV1Attr::get(
        elementsAttr.getContext(), vhloType, elementsAttr.getRawData());
  }

  LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << stablehloAttr << '\n');
  return {};  // Failed to convert attribute.
}

#undef RETURN_CONVERTED_ENUM_ATTR

template <typename StablehloOpTy>
class StablehloToVhloOpConverter : public OpConversionPattern<StablehloOpTy> {
 public:
  using OpConversionPattern<StablehloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      StablehloOpTy stablehloOp, typename StablehloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SmallVector<Type> vhloTypes;
    if (failed(this->getTypeConverter()->convertTypes(
            stablehloOp->getResultTypes(), vhloTypes))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed StableHLO -> VHLO type conversion\n");
      return failure();
    }

    // These operands have already been converted to VHLO by
    // the dialect conversion infrastructure.
    ValueRange vhloOperands = adaptor.getOperands();

    SmallVector<NamedAttribute> vhloAttrs;
    for (NamedAttribute stablehloAttr : stablehloOp->getAttrs()) {
      if constexpr (std::is_same<StablehloOpTy,
                                 stablehlo::CustomCallOp>::value) {
        if (stablehloAttr.getName() == "api_version") {
          auto vhloAttr = convertCustomCallApiVersion(stablehloAttr.getValue());
          if (!vhloAttr) return failure();
          vhloAttrs.push_back({stablehloAttr.getName(), vhloAttr});
          continue;
        }
      }
      auto vhloAttr =
          convertAttrToVhlo(stablehloAttr.getValue(), this->getTypeConverter());
      if (!vhloAttr) return failure();
      vhloAttrs.push_back({stablehloAttr.getName(), vhloAttr});
    }

    // Convert the vhlo operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // vhlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    StablehloToVhloOp<StablehloOpTy> vhloOp;
    if constexpr (std::is_same<StablehloOpTy, stablehlo::CaseOp>::value) {
      vhloOp = rewriter.replaceOpWithNewOp<vhlo::CaseOpV1>(
          stablehloOp, vhloTypes, vhloOperands, vhloAttrs,
          stablehloOp.getBranches().size());
    } else {
      vhloOp = rewriter.replaceOpWithNewOp<StablehloToVhloOp<StablehloOpTy>>(
          stablehloOp, vhloTypes, vhloOperands, vhloAttrs);
    }

    for (auto [stablehloRegion, vhloRegion] :
         llvm::zip(stablehloOp->getRegions(), vhloOp->getRegions())) {
      rewriter.inlineRegionBefore(stablehloRegion, vhloRegion,
                                  vhloRegion.end());
      if (failed(rewriter.convertRegionTypes(&vhloRegion,
                                             *this->getTypeConverter(),
                                             /*entryConversion=*/nullptr)))
        return failure();
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
    stablehlo::populateStablehloToVhloPatterns(&patterns, &converter,
                                               &getContext());
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
}  // namespace stablehlo
}  // namespace mlir
