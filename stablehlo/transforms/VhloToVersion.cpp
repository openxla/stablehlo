/* Copyright 2023 The StableHLO Authors.
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/VhloTypes.h"
#include "stablehlo/transforms/Passes.h"

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

// Currently there are no type-to-version conversions so this class
// simply validates that all types are from the VHLO dialect.
class VhloToVersionConverter : public TypeConverter {
 public:
  VhloToVersionConverter() : TypeConverter() {
    addConversion([](Type type) -> Type {
      if (type.getDialect().getNamespace() ==
          vhlo::VhloDialect::getDialectNamespace())
        return type;
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return {};
    });
  }
};

FailureOr<Version> parseTargetVersion(llvm::StringRef versionRef) {
  if (versionRef == "current") return Version::getCurrentVersion();
  return Version::fromString(versionRef);
}

// Check user-specified target version. Emit error if invalid.
FailureOr<Version> validateTargetVersion(llvm::StringRef versionRef,
                                         Operation* op) {
  auto failOrVersion = parseTargetVersion(versionRef);
  if (failed(failOrVersion)) {
    if (versionRef.empty())
      return emitError(op->getLoc())
             << "No target version specified.\n"
             << "Target version must be of the form #.#.# or 'current'.";
    return emitError(op->getLoc())
           << "Invalid target version argument '" << versionRef << "'\n"
           << "Target version must be of the form #.#.# or 'current'.";
  }

  Version targetVersion = *failOrVersion;
  if (targetVersion < Version::getMinimumVersion())
    return emitError(op->getLoc()) << "target version " << targetVersion
                                   << " is less than minimum supported "
                                   << Version::getMinimumVersion();
  if (Version::getCurrentVersion() < targetVersion)
    return emitError(op->getLoc()) << "target version " << targetVersion
                                   << " is greater than current version "
                                   << Version::getCurrentVersion();

  return targetVersion;
}

template <typename VersionedInterface>
bool isLegalVersion(VersionedInterface& interface, const Version& target) {
  return interface.getMinVersion() <= target &&
         target <= interface.getMaxVersion();
}

// Forward declare, isLegal(Type|Attribute) are mutually recursive
LogicalResult isLegalType(Type type, const Version& targetVersion);

LogicalResult isLegalAttribute(const Attribute& attr, Version targetVersion) {
  auto attrInterface = dyn_cast<VersionedAttrInterface>(attr);
  if (!attrInterface || !isLegalVersion(attrInterface, targetVersion)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to legalize attribute " << attr
                            << " to version " << targetVersion << '\n');
    return failure();
  }

  // Recursively check attrs if VHLO attr is a container
  if (auto arrAttr = attr.dyn_cast<ArrayV1Attr>())
    return success(llvm::all_of(arrAttr.getValue(), [&](Attribute ele) {
      return succeeded(isLegalAttribute(ele, targetVersion));
    }));
  if (auto arrAttr = attr.dyn_cast<DictionaryV1Attr>()) {
    return success(llvm::all_of(
        arrAttr.getValue(), [&](std::pair<Attribute, Attribute> entry) {
          return succeeded(isLegalAttribute(entry.first, targetVersion)) &&
                 succeeded(isLegalAttribute(entry.second, targetVersion));
        }));
  }
  if (auto floatAttr = attr.dyn_cast<FloatV1Attr>())
    return isLegalType(floatAttr.getType(), targetVersion);
  if (auto intAttr = attr.dyn_cast<IntegerV1Attr>())
    return isLegalType(intAttr.getType(), targetVersion);
  if (auto tensorAttr = attr.dyn_cast<TensorV1Attr>())
    return isLegalType(tensorAttr.getType(), targetVersion);
  if (auto typeAttr = attr.dyn_cast<TypeV1Attr>())
    return isLegalType(typeAttr.getValue(), targetVersion);

  // Is VHLO and valid version, success.
  return success();
}

LogicalResult isLegalType(Type type, const Version& targetVersion) {
  // All valid VHLO types must have versioned type interface.
  auto typeInterface = dyn_cast<VersionedTypeInterface>(type);
  if (!typeInterface || !isLegalVersion(typeInterface, targetVersion)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to legalize type " << type
                            << " to version " << targetVersion << '\n');
    return failure();
  }

  // Recursively check types if VHLO type is a container.
  if (auto complex = type.dyn_cast<ComplexV1Type>())
    return isLegalType(complex.getElementType(), targetVersion);
  if (auto func = type.dyn_cast<FunctionV1Type>()) {
    auto validateType = [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion));
    };
    return success(llvm::all_of(func.getInputs(), validateType) &&
                   llvm::all_of(func.getOutputs(), validateType));
  }
  if (auto ranked = type.dyn_cast<RankedTensorV1Type>()) {
    auto encoding = ranked.getEncoding();
    if (encoding && failed(isLegalAttribute(encoding, targetVersion)))
      return failure();
    return isLegalType(ranked.getElementType(), targetVersion);
  }
  if (auto tuple = type.dyn_cast<TupleV1Type>())
    return success(llvm::all_of(tuple.getTypes(), [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion));
    }));
  if (auto quant = type.dyn_cast<UniformQuantizedV1Type>())
    return success(
        succeeded(isLegalType(quant.getStorageType(), targetVersion)) &&
        succeeded(isLegalType(quant.getExpressedType(), targetVersion)));
  if (auto unranked = type.dyn_cast<UnrankedTensorV1Type>())
    return isLegalType(unranked.getElementType(), targetVersion);

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
  auto isLegalAttrFn = [&](const NamedAttribute& attr) {
    return succeeded(isLegalAttribute(attr.getValue(), targetVersion));
  };
  if (!llvm::all_of(op->getAttrs(), isLegalAttrFn)) return false;

  // Validate types
  auto isLegalTypeFn = [&](Type t) {
    return succeeded(isLegalType(t, targetVersion));
  };
  if (!llvm::all_of(op->getOperandTypes(), isLegalTypeFn) ||
      !llvm::all_of(op->getResultTypes(), isLegalTypeFn))
    return false;
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
    if (failed(failOrVersion)) return signalPassFailure();
    Version targetVersion = *failOrVersion;

    // An op is legal if the target version is in the ops `[min, max]`
    // supported version range.
    // Example:
    //   CustomCallV1 0.0.0 -> 0.0.x
    //   CustomCallV2 0.1.0 -> 0.4.x
    //   CustomCallV3 0.5.0 -> Current
    // Target Current (0.5.0):
    //   V3 legal    { Current  in [0.5.0, Current] }
    //   V2 illegal  { Current !in [0.1.0, 0.4.0] }
    //   V1 illegal  { Current !in [0.0.0, 0.0.0] }
    // Target 0.4.0:
    //   V3 illegal { 0.4.0 !in [0.5.0, Current] }
    //   V2 legal   { 0.4.0  in [0.1.0, 0.4.0] }
    //   V1 illegal { 0.4.0 !in [0.0.0, 0.0.0] }
    // Target 0.0.0:
    //   V3 illegal { 0.0.0 !in [0.5.0, Current] }
    //   V2 illegal { 0.1.0 !in [0.1.0, 0.4.0] }
    //   V1 legal   { 0.0.0  in [0.0.0, 0.1.0] }
    target.addDynamicallyLegalDialect<VhloDialect>(
        [&targetVersion](Operation* op) {
          return isLegalOperation(op, targetVersion);
        });
    target.addIllegalDialect<stablehlo::StablehloDialect, func::FuncDialect>();

    vhlo::VhloToVersionConverter converter;
    RewritePatternSet patterns(&getContext());
    stablehlo::populateVhloToVersionPatterns(&patterns, &converter,
                                             &getContext());

    // Conversions within VHLO may fail if new features or ops are used.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

////////////////////////////////////////////
/// Upgrade and Downgrade Infrastructure ///
////////////////////////////////////////////

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
    if (failed(prepareOpForConversion(op))) return failure();
    auto newOp = rewriter.replaceOpWithNewOp<TargetOp>(
        op, op->getResultTypes(), op->getOperands(), op->getAttrs());
    for (auto [oldRegion, newRegion] :
         llvm::zip(op->getRegions(), newOp->getRegions()))
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
    return success();
  }
};

/////////////////////////////////////////
/// Upgrade and Downgrade Definitions ///
/////////////////////////////////////////

}  // namespace
}  // namespace vhlo

namespace stablehlo {
void populateVhloToVersionPatterns(RewritePatternSet* patterns,
                                   TypeConverter* converter,
                                   MLIRContext* context) {
  // Currently empty because we're starting from a clean slate in v0.9.0.
}

}  // namespace stablehlo
}  // namespace mlir
