
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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/compatibility/dialect/VersionedStablehloOps.h"
#include "stablehlo/compatibility/transforms/CompatibilityTypeConversion.h"
#include "stablehlo/compatibility/transforms/MapStablehloToVersionedStablehlo.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace versionedhlo {
#define GEN_PASS_DEF_STABLEHLOLEGALIZETOVERSIONEDHLOPASS
#define GEN_PASS_DEF_VERSIONEDHLOLEGALIZETOSTABLEHLOPASS
#define GEN_PASS_DEF_VERSIONEDHLOUPGRADEPASS
#define GEN_PASS_DEF_VERSIONEDHLODOWNGRADEPASS
#define GEN_PASS_REGISTRATION
#include "stablehlo/compatibility/transforms/CompatibilityPasses.h.inc"

/// Registers all Torch transformation passes.
void registerStablehloCompatibilityPasses() { registerPasses(); }

namespace {
// FIXME: This should be removed when we have stablehlo.func
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

//////////////////////////////////
/// StableHLO --> VersionedHLO ///
//////////////////////////////////

#define RETURN_CONVERTED_ENUM_ATTR(Name)                             \
  auto stablehloValue = stablehlo::stringify##Name(attr.getValue()); \
  auto hloValue = versionedhlo::symbolize##Name(stablehloValue);     \
  if (!hloValue.has_value()) return {};                              \
  return versionedhlo::Name##Attr::get(attr.getContext(), hloValue.value())

Attribute convertAttrToVersionedhlo(Attribute stablehloAttr) {
  // Handle StableHLO attributes.
  // The logic that handles attributes from other dialects (e.g. builtin
  // attributes) lives below.
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::ChannelHandleAttr>()) {
    return versionedhlo::ChannelHandleAttr::get(
        attr.getContext(), attr.getHandle(), attr.getType());
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
    return versionedhlo::ConvDimensionNumbersAttr::get(
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
    return versionedhlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::FftTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType);
  }
  if (auto attr =
          stablehloAttr.dyn_cast<stablehlo::GatherDimensionNumbersAttr>()) {
    return versionedhlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr = stablehloAttr.dyn_cast<stablehlo::OutputOperandAliasAttr>()) {
    return versionedhlo::OutputOperandAliasAttr::get(
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
    return versionedhlo::ScatterDimensionNumbersAttr::get(
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
    // versionedhlo. This check is here only for exceptional situations, e.g.
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
      auto hloAttr = convertAttrToVersionedhlo(stablehloAttr);
      if (!hloAttr) return {};
      hloAttrs.push_back(hloAttr);
    }
    return ArrayAttr::get(stablehloAttrs.getContext(), hloAttrs);
  }
  return stablehloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

struct StablehloLegalizeToVersionedhloPass
    : public impl::StablehloLegalizeToVersionedhloPassBase<
          StablehloLegalizeToVersionedhloPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<versionedhlo::VersionedhloDialect>();

    versionedhlo::StablehloToVersionedhloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    versionedhlo::populateStablehloToVersionedhloPatterns(&patterns, &converter,
                                                          &getContext());
    registerFuncOpsForTypeConversion(target, patterns, converter);

    // StableHLO is a subset of VersionedHLO.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed partial conversion\n");
      // FIXME:
      // return signalPassFailure();
    }
  }
};

template <typename StablehloOpTy>
class StablehloToVersionedhloOpConverter
    : public OpConversionPattern<StablehloOpTy> {
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
      auto stablehloAttr = convertAttrToVersionedhlo(hloAttr.getValue());
      if (!stablehloAttr) return failure();
      stablehloAttrs.push_back({hloAttr.getName(), stablehloAttr});
    }

    // Convert the versionedhlo operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // versionedhlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    StablehloToVersionedhloOp<StablehloOpTy> versionedOp;
    if constexpr (std::is_same<StablehloOpTy, stablehlo::CaseOp>::value) {
      versionedOp = rewriter.replaceOpWithNewOp<versionedhlo::CaseOp>(
          stablehloOp, versionedTypes, stablehloOperands, stablehloAttrs,
          stablehloOp.getBranches().size());
    } else {
      versionedOp =
          rewriter.replaceOpWithNewOp<StablehloToVersionedhloOp<StablehloOpTy>>(
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
void populateStablehloToVersionedhloPatterns(RewritePatternSet* patterns,
                                             TypeConverter* converter,
                                             MLIRContext* context) {
  patterns->add<StablehloToVersionedhloOpConverter<StablehloOpTypes>...>(
      *converter, context);
}

//////////////////////////////////
/// VersionedHLO --> StableHLO ///
//////////////////////////////////
#define RETURN_CONVERTED_ENUM_ATTR(Name)                                \
  auto stablehloValue = versionedhlo::stringify##Name(attr.getValue()); \
  auto hloValue = stablehlo::symbolize##Name(stablehloValue);           \
  if (!hloValue.has_value()) return {};                                 \
  return stablehlo::Name##Attr::get(attr.getContext(), hloValue.value())

Attribute convertAttrToStablehlo(Attribute versionedhloAttr) {
  LLVM_DEBUG(llvm::dbgs() << "Converting " << versionedhloAttr);
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::ChannelHandleAttr>()) {
    return stablehlo::ChannelHandleAttr::get(attr.getContext(),
                                             attr.getHandle(), attr.getType());
  }
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::ComparisonDirectionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection);
  }
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::ComparisonTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonType);
  }
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::ConvDimensionNumbersAttr>()) {
    return stablehlo::ConvDimensionNumbersAttr::get(
        attr.getContext(), attr.getInputBatchDimension(),
        attr.getInputFeatureDimension(), attr.getInputSpatialDimensions(),
        attr.getKernelInputFeatureDimension(),
        attr.getKernelOutputFeatureDimension(),
        attr.getKernelSpatialDimensions(), attr.getOutputBatchDimension(),
        attr.getOutputFeatureDimension(), attr.getOutputSpatialDimensions());
  }
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::CustomCallApiVersionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion);
  }
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::DotDimensionNumbersAttr>()) {
    return stablehlo::DotDimensionNumbersAttr::get(
        attr.getContext(), attr.getLhsBatchingDimensions(),
        attr.getRhsBatchingDimensions(), attr.getLhsContractingDimensions(),
        attr.getRhsContractingDimensions());
  }
  if (auto attr = versionedhloAttr.dyn_cast<versionedhlo::FftTypeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(FftType);
  }
  if (auto attr = versionedhloAttr
                      .dyn_cast<versionedhlo::GatherDimensionNumbersAttr>()) {
    return stablehlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::OutputOperandAliasAttr>()) {
    return stablehlo::OutputOperandAliasAttr::get(
        attr.getContext(), attr.getOutputTupleIndices(), attr.getOperandIndex(),
        attr.getOperandTupleIndices());
  }
  if (auto attr = versionedhloAttr.dyn_cast<versionedhlo::PrecisionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Precision);
  }
  if (auto attr = versionedhloAttr.dyn_cast<versionedhlo::RngAlgorithmAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngAlgorithm);
  }
  if (auto attr =
          versionedhloAttr.dyn_cast<versionedhlo::RngDistributionAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(RngDistribution);
  }
  if (auto attr = versionedhloAttr
                      .dyn_cast<versionedhlo::ScatterDimensionNumbersAttr>()) {
    return stablehlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getScatterDimsToOperandDims(),
        attr.getIndexVectorDim());
  }
  if (auto attr = versionedhloAttr.dyn_cast<versionedhlo::TransposeAttr>()) {
    RETURN_CONVERTED_ENUM_ATTR(Transpose);
  }
  if (versionedhloAttr.getDialect().getNamespace() ==
      stablehlo::StablehloDialect::getDialectNamespace()) {
    // Our guiding principle is to support all versionedhlo functionality in
    // versionedhlo. This check is here only for exceptional situations, e.g.
    // when we added a new versionedhlo attribute and forgot to update the code
    // above.
    return {};
  }

  // Handle non-versionedhlo attributes.
  // If an attribute is not defined in versionedhlo, then it is unchanged,
  // with the exception of ArrayAttr which is converted recursively.
  if (auto versionedhloAttrs = versionedhloAttr.dyn_cast<ArrayAttr>()) {
    SmallVector<Attribute> hloAttrs;
    for (auto versionedhloAttr : versionedhloAttrs) {
      auto hloAttr = convertAttrToStablehlo(versionedhloAttr);
      if (!hloAttr) return {};
      hloAttrs.push_back(hloAttr);
    }
    return ArrayAttr::get(versionedhloAttrs.getContext(), hloAttrs);
  }
  return versionedhloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

struct VersionedhloLegalizeToStablehloPass
    : public impl::VersionedhloLegalizeToStablehloPassBase<
          VersionedhloLegalizeToStablehloPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<versionedhlo::VersionedhloDialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();

    versionedhlo::VersionedhloToStablehloTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    versionedhlo::populateVersionedhloToStablehloPatterns(&patterns, &converter,
                                                          &getContext());
    registerFuncOpsForTypeConversion(target, patterns, converter);

    // VersionedHLO should always be convertible to StableHLO if upgraded.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

template <typename VersionedhloOpTy>
class VersionedhloToStablehloOpConverter
    : public OpConversionPattern<VersionedhloOpTy> {
 public:
  using OpConversionPattern<VersionedhloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      VersionedhloOpTy stablehloOp, typename VersionedhloOpTy::Adaptor adaptor,
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

    // Convert the versionedhlo operation to a StableHLO equivalent.
    // This can almost be done in a generic fashion, except for
    // versionedhlo.case that uses a variadic number of regions which means an
    // additional argument for the generic builder.
    VersionedhloToStablehloOp<VersionedhloOpTy> versionedOp;
    if constexpr (std::is_same<VersionedhloOpTy, versionedhlo::CaseOp>::value) {
      versionedOp = rewriter.replaceOpWithNewOp<stablehlo::CaseOp>(
          stablehloOp, versionedTypes, stablehloOperands, stablehloAttrs,
          stablehloOp.getBranches().size());
    } else {
      versionedOp =
          rewriter
              .replaceOpWithNewOp<VersionedhloToStablehloOp<VersionedhloOpTy>>(
                  stablehloOp, versionedTypes, stablehloOperands,
                  stablehloAttrs);
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
void populateVersionedhloToStablehloPatterns(RewritePatternSet* patterns,
                                             TypeConverter* converter,
                                             MLIRContext* context) {
  patterns->add<VersionedhloToStablehloOpConverter<
      StablehloToVersionedhloOp<StablehloOpTypes>>...>(*converter, context);
}

}  // namespace

void populateStablehloToVersionedhloPatterns(RewritePatternSet* patterns,
                                             TypeConverter* converter,
                                             MLIRContext* context) {
  populateStablehloToVersionedhloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

void populateVersionedhloToStablehloPatterns(RewritePatternSet* patterns,
                                             TypeConverter* converter,
                                             MLIRContext* context) {
  populateVersionedhloToStablehloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

/////////////////////////////
/// VersionedHLO Upgrades ///
/////////////////////////////
namespace {
struct VersionedhloUpgradePass
    : public impl::VersionedhloUpgradePassBase<VersionedhloUpgradePass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<VersionedhloDialect>([](Operation* op) {
      if (auto versionInterface = dyn_cast<VersionInterface>(op)) {
        return versionInterface.isLatestVersion();
      }
      return false;
    });

    versionedhlo::VersionedTypeConverterBase converter;  // FIXME
    RewritePatternSet patterns(&getContext());
    versionedhlo::populateVersionedhloUpgradePatterns(&patterns, &converter,
                                                      &getContext());

    // Conversion from VersionedHLO to StableHLO should never fail.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

class CustomCallOpToV2 : public OpConversionPattern<CustomCallOp> {
 public:
  using OpConversionPattern<CustomCallOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      CustomCallOp stablehloOp, typename CustomCallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<CustomCallOpV2>(
        stablehloOp, stablehloOp->getResultTypes(), stablehloOp.getOperands(),
        stablehloOp->getAttrs());

    return success();
  }
};
}  // namespace

void populateVersionedhloUpgradePatterns(RewritePatternSet* patterns,
                                         TypeConverter* converter,
                                         MLIRContext* context) {
  patterns->add<CustomCallOpToV2>(*converter, context);
}

///////////////////////////////
/// VersionedHLO Downgrades ///
///////////////////////////////
namespace {
int64_t getMajorVersion(llvm::StringRef ref) {
  // Precondition: must be x.y.z
  auto isDot = [](char c){return c == '.';};
  auto majorS = ref.drop_until(isDot).drop_front(1).take_until(isDot);
  int64_t major;
  if (majorS.getAsInteger(/*radix=*/10, major)) {
    return -1;  // FIXME: Error
  }
  return major;
}

struct VersionedhloDowngradePass
    : public impl::VersionedhloDowngradePassBase<VersionedhloDowngradePass> {
  VersionedhloDowngradePass()
      : impl::VersionedhloDowngradePassBase<VersionedhloDowngradePass>() {}
  VersionedhloDowngradePass(VersionedhloDowngradePassOptions const& opts)
      : impl::VersionedhloDowngradePassBase<VersionedhloDowngradePass>(opts) {}

  void runOnOperation() override {
    ConversionTarget target(getContext());
    VersionedhloDowngradePassOptions opts{targetVersion};

    target.addDynamicallyLegalDialect<VersionedhloDialect>([&opts](Operation* op) {
      if (auto versionInterface = dyn_cast<VersionInterface>(op)) {
        // An op is legal if it's minimum supported version number is less than
        // or equal to the target version.
        // Example:
        //   CustomCallv1 0.0.0 -> 0.1.0
        //   CustomCallv2 0.1.0 -> 0.5.0
        //   CustomCallv3 0.5.0 -> <inf>
        // Target 0.4.0.
        //   v3 illegal { 0.5 > 0.4 }
        //   v2 legal   { 0.1 < 0.4 }
        // Target 0.0.0
        //   v3 illegal { 0.5 > 0.0 }
        //   v2 illegal { 0.1 > 0.0 }
        //   v1 legal   { 0.0 <= 0.0}
        return getMajorVersion(versionInterface.getMinVersion()) <=
               opts.targetVersion;
      }
      return false;
    });

    versionedhlo::VersionedTypeConverterBase converter;
    RewritePatternSet patterns(&getContext());
    versionedhlo::populateVersionedhloDowngradePatterns(&patterns, &converter,
                                                        &getContext(), opts);

    // Conversion from VersionedHLO to StableHLO should never fail.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

LogicalResult emitDowngradeError(Operation* op, llvm::StringRef message) {
  return op->emitError("failed to downgrade ") << op->getName() << ", " << message;
}

using VersionNumber = int64_t;

class CustomCallOpV1Downgrade : public OpConversionPattern<CustomCallOpV2> {
 public:
  using OpConversionPattern<CustomCallOpV2>::OpConversionPattern;

  // Downgrade CustomCallOp to v0.0.0
  static VersionNumber getDowngradeVersionNumber() {
    // FIXME: return CustomCallOp::getMinVersion() ??
    // -- Think about how to make this class easier going forward.
    // -- Transform from A -> B if max(B) < targetVersion?
    return 0;
  }

  LogicalResult matchAndRewrite(
      CustomCallOpV2 op, typename CustomCallOpV2::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (op.getResultLayouts().has_value()) {
      return emitDowngradeError(op, "has an non-empty result layout.");
    }

    // Remove attr and downgrade.
    if (op->hasAttr("result_layout")) op->removeAttr("result_layout");
    rewriter.replaceOpWithNewOp<CustomCallOp>(op, op->getResultTypes(),
                                              op.getOperands(), op->getAttrs());

    return success();
  }
};

template <typename DowngradePattern>
void maybeAddDowngradePass(RewritePatternSet* patterns,
                           TypeConverter* converter, MLIRContext* context,
                           int64_t targetVersion) {
  // Apply conversion pattern if target version is lower than what the downgrade
  // would make the op be.
  // FIXME: if (targetVersion <= DowngradePattern::TargetOp::getMaxVersion())
  // {...}
  if (targetVersion <= DowngradePattern::getDowngradeVersionNumber()) {
    patterns->add<CustomCallOpV1Downgrade>(*converter, context);
  }
}

}  // namespace

void populateVersionedhloDowngradePatterns(
    RewritePatternSet* patterns, TypeConverter* converter, MLIRContext* context,
    VersionedhloDowngradePassOptions const& opts){
  maybeAddDowngradePass<CustomCallOpV1Downgrade>(patterns, converter, context,
                                                 opts.targetVersion);
}

}  // namespace versionedhlo
}  // namespace mlir