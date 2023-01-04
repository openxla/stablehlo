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

#include <climits>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/TypeConversion.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace stablehlo {
#define GEN_PASS_DEF_VHLOTOVERSIONPASS
#include "stablehlo/transforms/Passes.h.inc"
}  // namespace stablehlo

///////////////////////
/// VHLO To Version ///
///////////////////////
namespace vhlo {
namespace {

FailureOr<Version> parseTargetVersion(llvm::StringRef versionRef) {
  if (versionRef == "current") {
    return VhloDialect::getCurrentVersion();
  }
  return Version::fromString(versionRef);
}

// Check user-specified target version. Emit error if invalid.
FailureOr<Version> validateTargetVersion(llvm::StringRef versionRef,
                                         Operation* op) {
  auto failOrVersion = parseTargetVersion(versionRef);
  if (failed(failOrVersion)) {
    if (versionRef.empty()) {
      return emitError(op->getLoc())
             << "No target version specified. Specify target using: "
                "--vhlo-to-version='target=[targetVersion]'\n"
             << "Target version must be of the form #.#.# or 'current'.";
    }
    return emitError(op->getLoc())
           << "Invalid target version argument '" << versionRef << "'\n"
           << "Target version must be of the form #.#.# or 'current'.";
  }

  Version targetVersion = *failOrVersion;
  if (targetVersion < VhloDialect::getMinimumVersion()) {
    return emitError(op->getLoc()) << "target version " << targetVersion
                                   << " is less than minimum supported "
                                   << VhloDialect::getMinimumVersion();
  }
  if (VhloDialect::getCurrentVersion() < targetVersion) {
    return emitError(op->getLoc()) << "target version " << targetVersion
                                   << " is greater than current version "
                                   << VhloDialect::getCurrentVersion();
  }
  return targetVersion;
}

template <typename VersionedInterface>
bool isLegalVersion(VersionedInterface& interface, const Version& target) {
  return interface.getMinVersion() <= target &&
         target <= interface.getMaxVersion();
}

LogicalResult isLegalAttribute(const Attribute& attr, Version targetVersion) {
  auto attrInterface = dyn_cast<VersionedAttrInterface>(attr);
  if (attrInterface && isLegalVersion(attrInterface, targetVersion)) {
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "failed to legalize attribute " << attr
                          << " to version " << targetVersion << '\n');
  return failure();
}

LogicalResult isLegalType(Type type, const Version& targetVersion) {
  // All valid VHLO types must have versioned type interface.
  auto typeInterface = dyn_cast<VersionedTypeInterface>(type);
  if (!typeInterface) {
    LLVM_DEBUG(llvm::dbgs() << "Unsupported type " << type << '\n');
    return failure();
  }

  // All valid VHLO types must satisfy the target version.
  if (!isLegalVersion(typeInterface, targetVersion)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to legalize type " << type
                            << " to version " << targetVersion << '\n');
    return failure();
  }

  // Recursively check types.
  if (auto ranked = type.dyn_cast<RankedTensorV1Type>()) {
    auto encoding = ranked.getEncoding();
    if (encoding && failed(isLegalAttribute(encoding, targetVersion)))
      return failure();
    return isLegalType(ranked.getElementType(), targetVersion);
  }
  if (auto unranked = type.dyn_cast<UnrankedTensorType>()) {
    return isLegalType(unranked.getElementType(), targetVersion);
  }
  if (auto tuple = type.dyn_cast<TupleType>()) {
    return success(llvm::all_of(tuple.getTypes(), [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion));
    }));
  }

  // Is VHLO and valid version, success.
  return success();
}

bool isLegalOperation(Operation* op, const Version& targetVersion) {
  // Validate op
  auto opInterface = dyn_cast<VersionedOpInterface>(op);
  if (!opInterface) return false;
  if (!isLegalVersion(opInterface, targetVersion)) return false;
  LLVM_DEBUG(llvm::dbgs() << "Legal version for target. " << op << '\n');

  // Validate attributes
  auto isLegalAttrFn = [&](NamedAttribute const& attr) {
    return succeeded(isLegalAttribute(attr.getValue(), targetVersion));
  };
  if (!llvm::all_of(op->getAttrs(), isLegalAttrFn)) return false;

  // Validate types
  auto isLegalTypeFn = [&](Type t) {
    return succeeded(isLegalType(t, targetVersion));
  };
  if (!llvm::all_of(op->getOperandTypes(), isLegalTypeFn) ||
      !llvm::all_of(op->getResultTypes(), isLegalTypeFn)) {
    return false;
  }

  return true;
}

using stablehlo::VhloToVersionPassOptions;
using stablehlo::impl::VhloToVersionPassBase;
struct VhloToVersionPass : public VhloToVersionPassBase<VhloToVersionPass> {
  VhloToVersionPass() : VhloToVersionPassBase<VhloToVersionPass>() {}
  VhloToVersionPass(const VhloToVersionPassOptions& opts)
      : VhloToVersionPassBase<VhloToVersionPass>(opts) {}

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // Validate version number
    auto failOrVersion =
        validateTargetVersion(targetVersionOption, getOperation());
    if (failed(failOrVersion)) {
      return signalPassFailure();
    }
    Version targetVersion = *failOrVersion;

    // An op is legal if the target version is in the ops `[min, max]`
    // supported version range.
    // Example:
    //   CustomCallV1 0.0.0 -> 0.0.x
    //   CustomCallV2 0.1.0 -> 0.4.x
    //   CustomCallV3 0.5.0 -> Curr
    // Target Curr (0.5.0):
    //   V3 legal    { Curr  in [0.5, Curr] }
    //   V2 illegal  { Curr !in [0.1, 0.4] }
    //   V1 illegal  { Curr !in [0.0, 0.0] }
    // Target 0.4.0:
    //   V3 illegal { 0.4 !in [0.5, Curr] }
    //   V2 legal   { 0.4  in [0.1, 0.4] }
    //   V1 illegal { 0.4 !in [0.0, 0.0] }
    // Target 0.0.0:
    //   V3 illegal { 0.0 !in [0.5, Curr] }
    //   V2 illegal { 0.1 !in [0.1, 0.4] }
    //   V1 legal   { 0.0  in [0.0, 0.1] }
    target.addDynamicallyLegalDialect<VhloDialect>(
        [&targetVersion](Operation* op) {
          return isLegalOperation(op, targetVersion);
        });

    vhlo::VhloToVersionConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateVhloToVersionPatterns(&patterns, &converter,
                                             &getContext());
    registerFuncOpsForTypeConversion(target, patterns, converter);

    // Conversions within VHLO may fail if new features or ops are used.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

////////////////////////////////////////////
/// Upgrade and Downgrade Infrastructure ///
////////////////////////////////////////////

LogicalResult emitDowngradeError(Operation* op, llvm::StringRef message) {
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

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    if (failed(prepareOpForConversion(op))) {
      return failure();
    }
    auto newOp = rewriter.replaceOpWithNewOp<TargetOp>(
        op, op->getResultTypes(), op->getOperands(), op->getAttrs());
    for (auto [oldRegion, newRegion] :
         llvm::zip(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
    }
    return success();
  }
};

/////////////////////////////////////////
/// Upgrade and Downgrade Definitions ///
/////////////////////////////////////////

// vhlo.custom_call --> vhlo.custom_call_v2
struct CustomCallOpV1ToV2
    : public VersionConversionPattern<CustomCallOpV1, CustomCallOpV2> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(CustomCallOpV1) const final {
    return success();
  }
};

// vhlo.custom_call_v2 --> vhlo.custom_call
struct CustomCallOpV2ToV1
    : public VersionConversionPattern<CustomCallOpV2, CustomCallOpV1> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(CustomCallOpV2 op) const final {
    if (op.getOutputOperandAliases()) {
      auto aliases =
          op.getOutputOperandAliases().value().dyn_cast<vhlo::ArrayV1Attr>();
      if (!aliases || !aliases.getValue().empty()) {
        return emitDowngradeError(
            op, "op has a non-empty output_operand_aliases attribute");
      }
      // Safe to downgrade.
      op->removeAttr("output_operand_aliases");
    }
    return success();
  }
};

// vhlo.collective_permute --> vhlo.collective_permute_v2
struct CollectivePermuteOpV1ToV2
    : public VersionConversionPattern<CollectivePermuteOpV1,
                                      CollectivePermuteOpV2> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(CollectivePermuteOpV1) const final {
    return success();
  }
};

// vhlo.collective_permute_v2 --> vhlo.collective_permute
struct CollectivePermuteOpV2ToV1
    : public VersionConversionPattern<CollectivePermuteOpV2,
                                      CollectivePermuteOpV1> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(CollectivePermuteOpV2 op) const final {
    if (op.getChannelHandle().has_value()) {
      return emitDowngradeError(op,
                                "op has a non-empty channel_handle attribute");
    }
    return success();
  }
};

// vhlo.all_gather--> vhlo.all_gather_v2
struct AllGatherOpV1ToV2
    : public VersionConversionPattern<AllGatherOpV1, AllGatherOpV2> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(AllGatherOpV1) const final {
    return success();
  }
};

// vhlo.all_gather_v2 --> vhlo.all_gather
struct AllGatherOpV2ToV1
    : public VersionConversionPattern<AllGatherOpV2, AllGatherOpV1> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(AllGatherOpV2 op) const final {
    if (op.getUseGlobalDeviceIdsAttr()) {
      return emitDowngradeError(
          op, "op has a non-empty use_global_device_ids attribute");
    }
    return success();
  }
};

// vhlo.all_to_all --> vhlo.all_to_all_v2
struct AllToAllOpV1ToV2
    : public VersionConversionPattern<AllToAllOpV1, AllToAllOpV2> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(AllToAllOpV1) const final {
    return success();
  }
};

// vhlo.all_to_all_v2 --> vhlo.all_to_all
struct AllToAllOpV2ToV1
    : public VersionConversionPattern<AllToAllOpV2, AllToAllOpV1> {
  using VersionConversionPattern::VersionConversionPattern;
  LogicalResult prepareOpForConversion(AllToAllOpV2 op) const final {
    if (op.getChannelHandle().has_value()) {
      return emitDowngradeError(op,
                                "op has a non-empty channel_handle attribute");
    }
    return success();
  }
};

}  // namespace
}  // namespace vhlo

namespace stablehlo {
void populateVhloToVersionPatterns(RewritePatternSet* patterns,
                                   TypeConverter* converter,
                                   MLIRContext* context) {
  patterns->add<vhlo::CustomCallOpV1ToV2>(*converter, context);
  patterns->add<vhlo::CustomCallOpV2ToV1>(*converter, context);
  patterns->add<vhlo::CollectivePermuteOpV1ToV2>(*converter, context);
  patterns->add<vhlo::CollectivePermuteOpV2ToV1>(*converter, context);
  patterns->add<vhlo::AllGatherOpV1ToV2>(*converter, context);
  patterns->add<vhlo::AllGatherOpV2ToV1>(*converter, context);
  patterns->add<vhlo::AllToAllOpV1ToV2>(*converter, context);
  patterns->add<vhlo::AllToAllOpV2ToV1>(*converter, context);
}

}  // namespace stablehlo
}  // namespace mlir
