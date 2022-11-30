
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

#include "stablehlo/compatibility/transforms/CompatibilityPasses.h"

#include <climits>
#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/compatibility/dialect/VersionNumber.h"
#include "stablehlo/compatibility/dialect/VhloOps.h"
#include "stablehlo/compatibility/transforms/CompatibilityTypeConversion.h"
#include "stablehlo/compatibility/transforms/MapStablehloToVhlo.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace vhlo {
#define GEN_PASS_DEF_STABLEHLOLEGALIZETOVHLOPASS
#define GEN_PASS_DEF_VHLOLEGALIZETOSTABLEHLOPASS
#define GEN_PASS_DEF_VHLOTOVERSIONPASS
#define GEN_PASS_REGISTRATION
#include "stablehlo/compatibility/transforms/CompatibilityPasses.h.inc"

/// Registers all Torch transformation passes.
void registerStablehloCompatibilityPasses() { registerPasses(); }

// From CompatibilityTypeConversion.h
void registerFuncOpsForTypeConversion(ConversionTarget& target,
                                      RewritePatternSet& patterns,
                                      TypeConverter& converter) {
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return converter.isSignatureLegal(op.getCalleeType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return converter.isLegal(op.getOperandTypes());
  });
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
}

namespace {

//////////////////////////
/// StableHLO --> VHLO ///
//////////////////////////

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
      versionedOp = rewriter.replaceOpWithNewOp<vhlo::CaseOp>(
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

//////////////////////////
/// VHLO --> StableHLO ///
//////////////////////////
#define RETURN_CONVERTED_ENUM_ATTR(Name)                        \
  auto stablehloValue = vhlo::stringify##Name(attr.getValue()); \
  auto hloValue = stablehlo::symbolize##Name(stablehloValue);   \
  if (!hloValue.has_value()) return {};                         \
  return stablehlo::Name##Attr::get(attr.getContext(), hloValue.value())

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
      stablehlo::StablehloDialect::getDialectNamespace()) {
    // Our guiding principle is to support all vhlo functionality in
    // vhlo. This check is here only for exceptional situations, e.g.
    // when we added a new vhlo attribute and forgot to update the code
    // above.
    return {};
  }

  // Handle non-vhlo attributes.
  // If an attribute is not defined in vhlo, then it is unchanged,
  // with the exception of ArrayAttr which is converted recursively.
  if (auto vhloAttrs = vhloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> hloAttrs;
    for (auto vhloAttr : vhloAttrs) {
      auto hloAttr = convertAttrToStablehlo(vhloAttr);
      if (!hloAttr) return {};
      hloAttrs.push_back(hloAttr);
    }
    return ArrayAttr::get(vhloAttrs.getContext(), hloAttrs);
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
    vhlo::populateVhloToStablehloPatterns(&patterns, &converter, &getContext());
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
      VhloOpTy stablehloOp, typename VhloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    SmallVector<Type> versionedTypes;
    if (failed(this->getTypeConverter()->convertTypes(
            stablehloOp->getResultTypes(), versionedTypes)))
      return failure();

    // These operands have already been converted to StableHLO by
    // the dialect conversion infrastructure.
    ValueRange stablehloOperands = adaptor.getOperands();

    SmallVector<NamedAttribute> stablehloAttrs;
    for (NamedAttribute hloAttr : stablehloOp->getAttrs()) {
      auto stablehloAttr = convertAttrToStablehlo(hloAttr.getValue());
      if (!stablehloAttr) return failure();
      stablehloAttrs.push_back({hloAttr.getName(), stablehloAttr});
    }

    // Convert the vhlo operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // vhlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    VhloToStablehloOp<VhloOpTy> versionedOp;
    if constexpr (std::is_same<VhloOpTy, vhlo::CaseOp>::value) {
      versionedOp = rewriter.replaceOpWithNewOp<stablehlo::CaseOp>(
          stablehloOp, versionedTypes, stablehloOperands, stablehloAttrs,
          stablehloOp.getBranches().size());
    } else {
      versionedOp = rewriter.replaceOpWithNewOp<VhloToStablehloOp<VhloOpTy>>(
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
void populateVhloToStablehloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  patterns
      ->add<VhloToStablehloOpConverter<StablehloToVhloOp<StablehloOpTypes>>...>(
          *converter, context);
}

}  // namespace

void populateStablehloToVhloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  populateStablehloToVhloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

void populateVhloToStablehloPatterns(RewritePatternSet* patterns,
                                     TypeConverter* converter,
                                     MLIRContext* context) {
  populateVhloToStablehloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

///////////////////////////////
/// VHLO To Version ///
///////////////////////////////
namespace {

struct VhloToVersionPass
    : public impl::VhloToVersionPassBase<VhloToVersionPass> {
  VhloToVersionPass() : impl::VhloToVersionPassBase<VhloToVersionPass>() {}
  VhloToVersionPass(VhloToVersionPassOptions const& opts)
      : impl::VhloToVersionPassBase<VhloToVersionPass>(opts) {}

  FailureOr<VersionNumber> validateTargetVersion(llvm::StringRef versionRef) {
    auto failOrVersion = VersionNumber::get(targetVersion);
    if (failed(failOrVersion)) {
      return emitError(
          getOperation()->getLoc(),
          "invalid target version number argument " + targetVersion);
    }
    VersionNumber targetVersionNumber = *failOrVersion;
    if (targetVersionNumber < VersionNumber::getMinimumSupported()) {
      return emitError(getOperation()->getLoc())
             << "target version " << targetVersion
             << " is less than minimum supported";
    }
    if (VersionNumber::getCurrent() < targetVersionNumber) {
      return emitError(getOperation()->getLoc())
             << "target version " << targetVersion
             << " is greater than current version";
    }
    return targetVersionNumber;
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // Validate version number
    auto failOrVersion = validateTargetVersion(targetVersion);
    if (failed(failOrVersion)) {
      return signalPassFailure();
    }
    VersionNumber targetVersionNumber = *failOrVersion;

    // An op is legal if the target version is in the ops `[min, max]`
    // supported version range.
    // Example:
    //   CustomCallv1 0.0.0 -> 0.0.x
    //   CustomCallv2 0.1.0 -> 0.4.x
    //   CustomCallv3 0.5.0 -> Curr
    // Target Curr (0.5.0):
    //   v3 legal    { Curr  in [0.5, Curr] }
    //   v2 illegal  { Curr !in [0.1, 0.4] }
    //   v1 illegal  { Curr !in [0.0, 0.0] }
    // Target 0.4.0:
    //   v3 illegal { 0.4 !in [0.5, Curr] }
    //   v2 legal   { 0.4  in [0.1, 0.4] }
    //   v1 illegal { 0.4 !in [0.0, 0.0] }
    // Target 0.0.0:
    //   v3 illegal { 0.0 !in [0.5, Curr] }
    //   v2 illegal { 0.1 !in [0.1, 0.4] }
    //   v1 legal   { 0.0  in [0.0, 0.1] }
    target.addDynamicallyLegalDialect<VhloDialect>(
        [&targetVersionNumber](Operation* op) {
          if (auto interface = dyn_cast<VersionInterface>(op)) {
            return (interface.getMinVersion() <= targetVersionNumber &&
                    targetVersionNumber <= interface.getMaxVersion());
          }
          return false;
        });

    vhlo::VersionedTypeConverterBase converter;
    RewritePatternSet patterns(&getContext());
    vhlo::populateVhloToVersionPatterns(&patterns, &converter, &getContext());

    // Conversion from VHLO to StableHLO should never fail.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

////////////////////////////////////////////
/// Upgrade and Downgrade Infrastructure ///
////////////////////////////////////////////

LogicalResult emitToVersionError(Operation* op, llvm::StringRef message) {
  return op->emitError("failed to downgrade ")
         << op->getName() << ", " << message;
}

template <typename SourceOp, typename TargetOp>
struct VersionConversionPattern : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  // This method allows subclasses to add or remove attributes if needed.
  // Can also fail if an op uses a feature that cannot be represented
  // in previous versions of the opset.
  virtual LogicalResult prepareOpForConversion(SourceOp op) const = 0;

  virtual LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (failed(prepareOpForConversion(op))) {
      return failure();
    }
    auto newOp = rewriter.replaceOpWithNewOp<TargetOp>(
        op, op->getResultTypes(), op.getOperands(), op->getAttrs());
    for (auto [hloRegion, stablehloRegion] :
         llvm::zip(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(hloRegion, stablehloRegion,
                                  stablehloRegion.end());
    }
    return success();
  }
};

/////////////////////////////////////////
/// Upgrade and Downgrade Definitions ///
/////////////////////////////////////////

// vhlo.custom_call --> vhlo.custom_call_v2
struct CustomCallOpV2Upgrade
    : public VersionConversionPattern<CustomCallOp, CustomCallOpV2> {
  using VersionConversionPattern<CustomCallOp,
                                 CustomCallOpV2>::VersionConversionPattern;
  LogicalResult prepareOpForConversion(CustomCallOp) const final {
    return success();
  }
};

// vhlo.custom_call_v2 --> vhlo.custom_call
struct CustomCallOpV1Downgrade
    : public VersionConversionPattern<CustomCallOpV2, CustomCallOp> {
  using VersionConversionPattern<CustomCallOpV2,
                                 CustomCallOp>::VersionConversionPattern;
  LogicalResult prepareOpForConversion(CustomCallOpV2 op) const final {
    if (op.getResultLayouts().has_value()) {
      return emitToVersionError(op,
                                "op has a non-empty result_layouts attribute");
    }
    return success();
  }
};

}  // namespace

void populateVhloToVersionPatterns(RewritePatternSet* patterns,
                                   TypeConverter* converter,
                                   MLIRContext* context) {
  patterns->add<CustomCallOpV2Upgrade>(*converter, context);
  patterns->add<CustomCallOpV1Downgrade>(*converter, context);
}

}  // namespace vhlo
}  // namespace mlir