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

#include "stablehlo/transforms/Passes.h"

#include <climits>
#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/TypeConversion.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace vhlo {
#define GEN_PASS_DEF_VHLOTOVERSIONPASS
#include "stablehlo/transforms/Passes.h.inc"

///////////////////////
/// VHLO To Version ///
///////////////////////

namespace {

struct VhloToVersionPass
    : public impl::VhloToVersionPassBase<VhloToVersionPass> {
  VhloToVersionPass() : impl::VhloToVersionPassBase<VhloToVersionPass>() {}
  VhloToVersionPass(VhloToVersionPassOptions const& opts)
      : impl::VhloToVersionPassBase<VhloToVersionPass>(opts) {}

  FailureOr<Version> parseTargetVersion(llvm::StringRef versionRef) {
    if (versionRef == "current") {
      return VhloDialect::getCurrentDialectVersion();
    }
    return Version::get(versionRef);
  }

  FailureOr<Version> validateTargetVersion(llvm::StringRef versionRef) {
    auto failOrVersion = parseTargetVersion(versionRef);
    if (failed(failOrVersion)) {
      if (targetVersion.empty()) {
        return emitError(getOperation()->getLoc())
               << "No target version specified. Specify target using: --vhlo-to-version='target=[targetVersion]'\n"
               << "Target version must be of the form #.#.# or 'current'.";
      }
      return emitError(getOperation()->getLoc())
             << "Invalid target version number argument '" << targetVersion
             << "'\n"
             << "Target version must be of the form #.#.# or 'current'.";
    }
    
    Version targetVersion = *failOrVersion;
    if (targetVersion < VhloDialect::getMinimumDialectVersion()) {
      return emitError(getOperation()->getLoc())
             << "target version " << targetVersion.getAsArray()
             << " is less than minimum supported";
    }
    if (VhloDialect::getCurrentDialectVersion() < targetVersion) {
      return emitError(getOperation()->getLoc())
             << "target version " << targetVersion.getAsArray()
             << " is greater than current version";
    }
    return targetVersion;
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // Validate version number
    auto failOrVersion = validateTargetVersion(targetVersion);
    if (failed(failOrVersion)) {
      return signalPassFailure();
    }
    Version targetVersion = *failOrVersion;

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
        [&targetVersion](Operation* op) {
          if (auto interface = dyn_cast<VersionedInterface>(op)) {
            return (interface.getMinVersion() <= targetVersion &&
                    targetVersion <= interface.getMaxVersion());
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
    if (op.getResultLayouts().has_value()) {
      return emitDowngradeError(op,
                                "op has a non-empty result_layouts attribute");
    }
    return success();
  }
};

}  // namespace

void populateVhloToVersionPatterns(RewritePatternSet* patterns,
                                   TypeConverter* converter,
                                   MLIRContext* context) {
  patterns->add<CustomCallOpV1ToV2>(*converter, context);
  patterns->add<CustomCallOpV2ToV1>(*converter, context);
}

}  // namespace vhlo
}  // namespace mlir