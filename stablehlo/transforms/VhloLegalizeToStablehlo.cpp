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
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/VhloTypes.h"
#include "stablehlo/transforms/MapStablehloToVhlo.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/TypeConversion.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_VHLOLEGALIZETOSTABLEHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

///////////////////////////////////////////////
/// VHLO --> StableHLO Types and Attributes ///
///////////////////////////////////////////////

class VhloToStablehloTypeConverter : public vhlo::VhloTypeConverter {
 public:
  VhloToStablehloTypeConverter() : vhlo::VhloTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](vhlo::TokenV1Type token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return stablehlo::TokenType::get(token.getContext());
    });
    addVhloToBuiltinConversions();
  }

  Attribute convertEncoding(Attribute attr) final {
    if (auto vhloAttr = attr.dyn_cast_or_null<vhlo::TypeExtensionsV1Attr>()) {
      return stablehlo::TypeExtensionsAttr::get(vhloAttr.getContext(),
                                                vhloAttr.getBounds());
    }
    // All encodings supported in StableHLO.
    return attr;
  }
};

#define RETURN_CONVERTED_ENUM_ATTR(Name, Version)                   \
  auto vhloValue = vhlo::stringify##Name##Version(attr.getValue()); \
  auto stablehloValue = stablehlo::symbolize##Name(vhloValue);      \
  if (!stablehloValue.has_value()) return {};                       \
  return stablehlo::Name##Attr::get(attr.getContext(), stablehloValue.value())

Attribute convertAttrToStablehlo(Attribute vhloAttr,
                                 TypeConverter* typeConverter) {
  LLVM_DEBUG(llvm::dbgs() << "Converting attr " << vhloAttr);
  if (auto attr = vhloAttr.dyn_cast<vhlo::ChannelHandleV1Attr>()) {
    return stablehlo::ChannelHandleAttr::get(attr.getContext(),
                                             attr.getHandle(), attr.getType());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ComparisonDirectionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ComparisonTypeV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ConvDimensionNumbersV1Attr>()) {
    return stablehlo::ConvDimensionNumbersAttr::get(
        attr.getContext(), attr.getInputBatchDimension(),
        attr.getInputFeatureDimension(), attr.getInputSpatialDimensions(),
        attr.getKernelInputFeatureDimension(),
        attr.getKernelOutputFeatureDimension(),
        attr.getKernelSpatialDimensions(), attr.getOutputBatchDimension(),
        attr.getOutputFeatureDimension(), attr.getOutputSpatialDimensions());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::CustomCallApiVersionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::DotDimensionNumbersV1Attr>()) {
    return stablehlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::FftTypeV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::GatherDimensionNumbersV1Attr>()) {
    return stablehlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::OutputOperandAliasV1Attr>()) {
    return stablehlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::PrecisionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Precision, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::RngAlgorithmV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::RngDistributionV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution, V1);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::ScatterDimensionNumbersV1Attr>()) {
    return stablehlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getScatterDimsToOperandDims(),
        attr.getIndexVectorDim());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::TransposeV1Attr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose, V1);
  }

  // Forked attributes
  if (auto vhloAttrs = vhloAttr.dyn_cast<vhlo::ArrayV1Attr>()) {
    SmallVector<Attribute> stablehloAttrs;
    for (auto vhloAttr : vhloAttrs.getValue()) {
      auto stablehloAttr = convertAttrToStablehlo(vhloAttr, typeConverter);
      if (!stablehloAttr) return {};
      stablehloAttrs.push_back(stablehloAttr);
    }
    return ArrayAttr::get(vhloAttrs.getContext(), stablehloAttrs);
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::DenseIntOrFPElementsV1Attr>()) {
    auto builtinType = typeConverter->convertType(attr.getType());
    if (!builtinType) return {};
    return DenseIntOrFPElementsAttr::getFromRawBuffer(builtinType,
                                                      attr.getRawData());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::FlatSymbolRefV1Attr>()) {
    auto builtinRootRef =
        convertAttrToStablehlo(attr.getRootReference(), typeConverter);
    if (!builtinRootRef || !builtinRootRef.isa<StringAttr>()) return {};
    return FlatSymbolRefAttr::get(attr.getContext(),
                                  builtinRootRef.cast<StringAttr>());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::FloatV1Attr>()) {
    auto builtinFloatType = typeConverter->convertType(attr.getType());
    if (!builtinFloatType) return {};
    // FIXME: What is the proper way to reconstruct a attr?
    return FloatAttr::get(builtinFloatType, attr.getValue().convertToDouble());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::IntegerV1Attr>()) {
    auto builtinIntegerType = typeConverter->convertType(attr.getType());
    if (!builtinIntegerType) return {};
    return IntegerAttr::get(builtinIntegerType, attr.getValue());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::StringV1Attr>()) {
    return StringAttr::get(attr.getContext(), attr.getValue());
  }
  if (auto attr = vhloAttr.dyn_cast<vhlo::UnitV1Attr>()) {
    return UnitAttr::get(attr.getContext());
  }

  // All VHLO Attributes must be converted by now.
  if (vhloAttr.getDialect().getNamespace() ==
      vhlo::VhloDialect::getDialectNamespace()) {
    // All VHLO attributes must have counterparts in StableHLO.
    return {};
  }

  // This should be unreachable unless program is a mix of VHLO and other
  // due to user edits to textual assembly format.
  return {};
}

#undef RETURN_CONVERTED_ENUM_ATTR

/////////////////////////////////////
/// VHLO --> StableHLO Operations ///
/////////////////////////////////////

struct VhloLegalizeToStablehloPass
    : public impl::VhloLegalizeToStablehloPassBase<
          VhloLegalizeToStablehloPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<vhlo::VhloDialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();

    VhloToStablehloTypeConverter converter;
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
      auto stablehloAttr =
          convertAttrToStablehlo(vhloAttr.getValue(), this->getTypeConverter());
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
      if (failed(rewriter.convertRegionTypes(&stablehloRegion,
                                             *this->getTypeConverter(),
                                             /*entryConversion=*/nullptr)))
        return failure();
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
