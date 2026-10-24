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

#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/VhloTypes.h"
#include "stablehlo/transforms/Passes.h"

#define DEBUG_TYPE "stablehlo-compat"

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
class VhloToVersionConverter : public VhloTypeConverter {
 public:
  // Safe to allow other dialects since the StableHLO->VHLO conversion guards
  // the legality of types / attributes.
  VhloToVersionConverter() : VhloTypeConverter(/*allowOtherDialects=*/true) {
    addConversion([&](Type type) -> Type {
      if (llvm::isa<vhlo::VhloDialect>(type.getDialect())) {
        // Once a type becomes versioned we will check the version here.
        return type;
      }

      LLVM_DEBUG(llvm::dbgs() << "Not VHLO type, allowed: " << type << '\n');
      return type;
    });
  }

  Attribute convertEncoding(Attribute attr) const override { return attr; };
};

// Check user-specified target version. Emit error if invalid.
FailureOr<Version> validateTargetVersion(llvm::StringRef versionRef,
                                         Operation* op) {
  auto failOrVersion = Version::fromString(versionRef);
  if (failed(failOrVersion)) {
    if (versionRef.empty())
      return emitError(op->getLoc())
             << "No target version specified.\n"
             << "Target version must be of the form `#.#.#`.";
    return emitError(op->getLoc())
           << "Invalid target version argument '" << versionRef << "'\n"
           << "Target version must be of the form `#.#.#`.";
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

  // Opset changes warrant a minor version bump, so this conversion assumes
  // patch v0 since it is written against the opset at version `X.Y.0`.
  if (targetVersion.getPatch() != 0) {
    targetVersion =
        vhlo::Version(targetVersion.getMajor(), targetVersion.getMinor(), 0);
  }

  return targetVersion;
}

template <typename VersionedInterface>
bool isLegalVersion(VersionedInterface& interface, const Version& target) {
  return interface.getMinVersion() <= target &&
         target <= interface.getMaxVersion();
}

// The conversion framework may query legality more than once; report each
// distinct (location, type) failure only once.
using ReportedTypeFailures = llvm::DenseSet<std::pair<Location, Type>>;

// Forward declare, isLegal(Type|Attribute) are mutually recursive
LogicalResult isLegalType(Type type, const Version& targetVersion,
                          std::optional<Location> loc,
                          ReportedTypeFailures* reported);

LogicalResult isLegalAttribute(const Attribute& attr, Version targetVersion,
                               std::optional<Location> loc,
                               ReportedTypeFailures* reported) {
  // StableHLO->VHLO conversion guards the legality of types / attributes.
  // Other dialect types that make it past that step are valid here.
  if (!mlir::isa<vhlo::VhloDialect>(attr.getDialect())) return success();

  auto attrInterface = dyn_cast<VersionedAttrInterface>(attr);
  if (!attrInterface || !isLegalVersion(attrInterface, targetVersion)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to legalize attribute " << attr
                            << " to version " << targetVersion << '\n');
    return failure();
  }

  // Recursively check attrs if VHLO attr is a container
  if (auto arrAttr = dyn_cast<ArrayV1Attr>(attr)) {
    return success(llvm::all_of(arrAttr.getValue(), [&](Attribute ele) {
      return succeeded(isLegalAttribute(ele, targetVersion, loc, reported));
    }));
  }
  if (auto arrAttr = dyn_cast<DictionaryV1Attr>(attr)) {
    return success(llvm::all_of(
        arrAttr.getValue(), [&](std::pair<Attribute, Attribute> entry) {
          return succeeded(isLegalAttribute(entry.first, targetVersion, loc,
                                            reported)) &&
                 succeeded(isLegalAttribute(entry.second, targetVersion, loc,
                                            reported));
        }));
  }
  if (auto floatAttr = dyn_cast<FloatV1Attr>(attr))
    return isLegalType(floatAttr.getType(), targetVersion, loc, reported);
  if (auto intAttr = dyn_cast<IntegerV1Attr>(attr))
    return isLegalType(intAttr.getType(), targetVersion, loc, reported);
  if (auto tensorAttr = dyn_cast<TensorV1Attr>(attr))
    return isLegalType(tensorAttr.getType(), targetVersion, loc, reported);
  if (auto typeAttr = dyn_cast<TypeV1Attr>(attr))
    return isLegalType(typeAttr.getValue(), targetVersion, loc, reported);
  if (auto resultAccuracyAttr = dyn_cast<ResultAccuracyV1Attr>(attr))
    return isLegalAttribute(resultAccuracyAttr.getMode(), targetVersion, loc,
                            reported);
  if (auto axisRefAttr = dyn_cast<AxisRefV1Attr>(attr)) {
    return success(succeeded(isLegalAttribute(axisRefAttr.getName(),
                                              targetVersion, loc, reported)) &&
                   (!axisRefAttr.getSubAxisInfo() ||
                    succeeded(isLegalAttribute(axisRefAttr.getSubAxisInfo(),
                                               targetVersion, loc, reported))));
  }
  if (auto meshAxesAttr = dyn_cast<ReplicaGroupMeshAxesV1Attr>(attr)) {
    return success(succeeded(isLegalAttribute(meshAxesAttr.getMesh(),
                                              targetVersion, loc, reported)) &&
                   succeeded(isLegalAttribute(meshAxesAttr.getAxes(),
                                              targetVersion, loc, reported)));
  }

  // Is VHLO and valid version, success.
  return success();
}

LogicalResult isLegalType(Type type, const Version& targetVersion,
                          std::optional<Location> loc,
                          ReportedTypeFailures* reported) {
  // StableHLO->VHLO conversion guards the legality of types / attributes.
  // Other dialect types that make it past that step are valid here.
  if (!mlir::isa<vhlo::VhloDialect>(type.getDialect())) return success();

  // All valid VHLO types must have versioned type interface.
  auto typeInterface = dyn_cast<VersionedTypeInterface>(type);
  if (!typeInterface || !isLegalVersion(typeInterface, targetVersion)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to legalize type " << type
                            << " to version " << targetVersion << '\n');
    return failure();
  }

  // Check versioned type constraints (e.g. complex component types).
  if (auto constraint = dyn_cast<VersionedTypeConstraintInterface>(type)) {
    if (failed(constraint.validateConstraint(targetVersion))) {
      if (loc && reported && reported->insert({*loc, type}).second)
        return emitError(*loc) << "failed to legalize type " << type
                               << " for target version " << targetVersion;
      return failure();
    }
  }

  // Recursively check types if VHLO type is a container.
  if (auto complex = dyn_cast<ComplexV1Type>(type))
    return isLegalType(complex.getElementType(), targetVersion, loc, reported);
  if (auto func = dyn_cast<FunctionV1Type>(type)) {
    auto validateType = [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion, loc, reported));
    };
    return success(llvm::all_of(func.getInputs(), validateType) &&
                   llvm::all_of(func.getOutputs(), validateType));
  }
  if (auto ranked = dyn_cast<RankedTensorV1Type>(type)) {
    auto encoding = ranked.getEncoding();
    if (encoding &&
        failed(isLegalAttribute(encoding, targetVersion, loc, reported)))
      return failure();
    return isLegalType(ranked.getElementType(), targetVersion, loc, reported);
  }
  if (auto tuple = dyn_cast<TupleV1Type>(type)) {
    return success(llvm::all_of(tuple.getTypes(), [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion, loc, reported));
    }));
  }
  if (auto quant = dyn_cast<UniformQuantizedV1Type>(type))
    return success(succeeded(isLegalType(quant.getStorageType(), targetVersion,
                                         loc, reported)) &&
                   succeeded(isLegalType(quant.getExpressedType(),
                                         targetVersion, loc, reported)));
  if (auto unranked = dyn_cast<UnrankedTensorV1Type>(type))
    return isLegalType(unranked.getElementType(), targetVersion, loc, reported);

  if (auto buffer = dyn_cast<RankedBufferV1Type>(type)) {
    return isLegalType(buffer.getElementType(), targetVersion, loc, reported);
  }
  if (auto future = dyn_cast<FutureV1Type>(type))
    return success(llvm::all_of(future.getTypes(), [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion, loc, reported));
    }));
  if (auto quantPerAxis = dyn_cast<UniformQuantizedPerAxisV1Type>(type))
    return success(succeeded(isLegalType(quantPerAxis.getStorageType(),
                                         targetVersion, loc, reported)) &&
                   succeeded(isLegalType(quantPerAxis.getExpressedType(),
                                         targetVersion, loc, reported)));

  // Is VHLO and valid version, success.
  return success();
}

bool isLegalLocation(Location loc, const Version& targetVersion) {
  // FileLineColRange locations are a forward incompatibility in upstream MLIR
  // just before v1.8.4 was tagged. Conservatively use 1.9.0 since StableHLO
  // passes require major versions for incompats.
  //
  // Support for downgrading these locations exists in
  // StablehloCompatibilityExpanderPass.
  bool isLegal = true;
  loc->walk([&](Location childLoc) -> WalkResult {
    if (auto fileLineColLoc = dyn_cast<FileLineColRange>(childLoc)) {
      static const Version kFileLineColLocMinVersion = Version(1, 9, 0);
      if (!isStrictFileLineColLoc(childLoc) &&
          targetVersion < kFileLineColLocMinVersion) {
        LLVM_DEBUG(llvm::dbgs() << "failed to legalize location " << childLoc
                                << " to version " << targetVersion << '\n');
        isLegal = false;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return isLegal;
}

bool isLegalOperation(Operation* op, const Version& targetVersion,
                      ReportedTypeFailures& reported) {
  // Validate op
  auto opInterface = dyn_cast<VersionedOpInterface>(op);
  if (!opInterface) return true;
  if (!isLegalVersion(opInterface, targetVersion)) return false;
  LLVM_DEBUG(llvm::dbgs() << "Legal op version for target. " << op << '\n');

  // Validate op constraints
  auto constraintInterface = dyn_cast<VersionedOpConstraintInterface>(op);
  if (constraintInterface &&
      failed(constraintInterface.validateConstraint(op, targetVersion))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Op failed to satisfy versioned constraints. " << op << '\n');
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "Legal constraints for target. " << op << '\n');

  // Validate attributes
  auto isLegalAttrFn = [&](const NamedAttribute& attr) {
    return succeeded(isLegalAttribute(attr.getValue(), targetVersion,
                                      op->getLoc(), &reported));
  };
  if (!llvm::all_of(op->getAttrs(), isLegalAttrFn)) return false;
  LLVM_DEBUG(llvm::dbgs() << "Legal op attributes for target. " << op << '\n');

  // Validate types
  auto isLegalTypeFn = [&](Type t) {
    return succeeded(isLegalType(t, targetVersion, op->getLoc(), &reported));
  };
  if (!llvm::all_of(op->getOperandTypes(), isLegalTypeFn) ||
      !llvm::all_of(op->getResultTypes(), isLegalTypeFn))
    return false;
  LLVM_DEBUG(llvm::dbgs() << "Legal op types for target. " << op << '\n');

  // Validate region block argument types
  for (auto& region : op->getRegions())
    for (auto& block : region.getBlocks())
      if (!llvm::all_of(block.getArgumentTypes(), isLegalTypeFn)) return false;

  // Validate location
  LLVM_DEBUG(
      if (!isLegalLocation(op->getLoc(), targetVersion)) {
        llvm::dbgs() << "Illegal op location for target. " << op << '\n';
        // TODO: One integrated into ecosystem return an error in these cases.
      } else {
        llvm::dbgs() << "Legal op location for target. " << op << '\n';
      });

  return true;
}

using stablehlo::VhloToVersionPassOptions;
using stablehlo::impl::VhloToVersionPassBase;
struct VhloToVersionPass : public VhloToVersionPassBase<VhloToVersionPass> {
  VhloToVersionPass() : VhloToVersionPassBase<VhloToVersionPass>() {}
  VhloToVersionPass(const VhloToVersionPassOptions& opts)
      : VhloToVersionPassBase<VhloToVersionPass>(opts) {}

  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    stablehlo::populateVhloToVersionPatterns(context, &patterns_, &converter);
    patterns = std::move(patterns_);

    return success();
  }

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
    ReportedTypeFailures reported;
    target.addDynamicallyLegalDialect<VhloDialect>(
        [&targetVersion, &reported](Operation* op) {
          return isLegalOperation(op, targetVersion, reported);
        });

    // Conversions within VHLO may fail if new features or ops are used.
    if (failed(applyPartialConversion(getOperation(), target, patterns))) {
      getOperation()->emitError()
          << "failed to convert VHLO to v" << targetVersion;
      return signalPassFailure();
    }
  }

 private:
  vhlo::VhloToVersionConverter converter;
  FrozenRewritePatternSet patterns;
};

/////////////////////////////////////////
/// Upgrade and Downgrade Definitions ///
/////////////////////////////////////////

void copyDiscardableAttrs(Operation* src, Operation* dst) {
  dst->setDiscardableAttrs(src->getDiscardableAttrDictionary());
}

void copyDiscardableAttrs(Value src, Value dst) {
  copyDiscardableAttrs(src.getDefiningOp(), dst.getDefiningOp());
}

TensorV1Attr getEmptyI64Tensor(OpBuilder& builder) {
  auto shape = vhlo::RankedTensorV1Type::get(
      builder.getContext(), {0},
      vhlo::IntegerSI64V1Type::get(builder.getContext()), {});
  auto denseElements = DenseIntElementsAttr::get(
      RankedTensorType::get({0}, builder.getI64Type()),
      llvm::ArrayRef<int64_t>{});
  return vhlo::TensorV1Attr::get(builder.getContext(), shape, denseElements);
}

bool isEmptyTensor(Attribute attr) {
  auto tensor = dyn_cast<TensorV1Attr>(attr);
  if (tensor) return tensor.getData().getRawData().empty();
  return false;
}

ArrayV1Attr getEmptyArray(OpBuilder& builder) {
  return vhlo::ArrayV1Attr::get(builder.getContext(), {});
}

bool isEmptyArray(Attribute attr) {
  auto arrayAttr = dyn_cast<ArrayV1Attr>(attr);
  if (arrayAttr) return arrayAttr.getValue().empty();
  return false;
}

bool isNoneType(Attribute attr) {
  auto typeAttr = llvm::dyn_cast<TypeV1Attr>(attr);
  if (!typeAttr) return false;
  return isa<NoneV1Type>(typeAttr.getValue());
}

TypeV1Attr getNoneType(OpBuilder& builder) {
  return TypeV1Attr::get(builder.getContext(),
                         NoneV1Type::get(builder.getContext()));
}

TensorV1Attr getDefaultConvPadding(OpBuilder& builder, Value lhs) {
  auto lhsType = dyn_cast<RankedTensorV1Type>(lhs.getType());
  if (!lhsType) return TensorV1Attr();

  // Convert to DenseElements for getRawData handling.
  SmallVector<int64_t> paddingShape{
      static_cast<int64_t>(lhsType.getShape().size() - 2), 2};
  auto denseElements = DenseIntElementsAttr::get(
      RankedTensorType::get(paddingShape, builder.getI64Type()),
      SmallVector<int64_t>(paddingShape[0] * 2, 0ll));

  return TensorV1Attr::get(
      builder.getContext(),
      RankedTensorV1Type::get(builder.getContext(), paddingShape,
                              IntegerSI64V1Type::get(builder.getContext()),
                              nullptr),
      denseElements);
}

bool isFalseBoolean(Attribute attr) {
  auto boolAttr = dyn_cast<BooleanV1Attr>(attr);
  return boolAttr && !boolAttr.getValue();
}

BooleanV1Attr getFalseBoolean(OpBuilder& builder) {
  return BooleanV1Attr::get(builder.getContext(), false);
}

bool isDefaultResultAccuracy(Attribute attr) {
  auto resultAccuracy = dyn_cast<ResultAccuracyV1Attr>(attr);
  auto default_mode = ResultAccuracyModeV1Attr::get(
      attr.getContext(), ResultAccuracyModeV1::DEFAULT);
  return resultAccuracy.getAtol().isZero() &&
         resultAccuracy.getRtol().isZero() && resultAccuracy.getUlps() == 0 &&
         resultAccuracy.getMode() == default_mode;
}

ResultAccuracyV1Attr getDefaultResultAccuracy(OpBuilder& builder) {
  return ResultAccuracyV1Attr::get(
      builder.getContext(), APFloat(0.0), APFloat(0.0), 0,
      ResultAccuracyModeV1Attr::get(builder.getContext(),
                                    ResultAccuracyModeV1::DEFAULT));
}

// DRR has limited support for ops with regions
struct ScatterOpV2ToV1 : public OpRewritePattern<ScatterOpV2> {
  using OpRewritePattern<ScatterOpV2>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOpV2 op,
                                PatternRewriter& rewriter) const override {
    if (!isEmptyTensor(op.getScatterIndicesBatchingDims()) ||
        !isEmptyTensor(op.getInputBatchingDims())) {
      return rewriter.notifyMatchFailure(op, "non-empty batching dims");
    }
    auto newOp = rewriter.replaceOpWithNewOp<ScatterOpV1>(
        op, op->getResultTypes(), op.getInputs(), op.getScatterIndices(),
        op.getUpdates(), op.getUpdateWindowDims(), op.getInsertedWindowDims(),
        op.getScatterDimsToOperandDims(), op.getIndexVectorDim(),
        op.getIndicesAreSorted(), op.getUniqueIndices());
    Region& body = newOp.getUpdateComputation();
    rewriter.inlineRegionBefore(op.getUpdateComputation(), body, body.begin());
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

struct ScatterOpV1ToV2 : public OpRewritePattern<ScatterOpV1> {
  using OpRewritePattern<ScatterOpV1>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOpV1 op,
                                PatternRewriter& rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<ScatterOpV2>(
        op, op->getResultTypes(), op.getInputs(), op.getScatterIndices(),
        op.getUpdates(), op.getUpdateWindowDims(), op.getInsertedWindowDims(),
        getEmptyI64Tensor(rewriter), getEmptyI64Tensor(rewriter),
        op.getScatterDimsToOperandDims(), op.getIndexVectorDim(),
        op.getIndicesAreSorted(), op.getUniqueIndices());
    Region& body = newOp.getUpdateComputation();
    rewriter.inlineRegionBefore(op.getUpdateComputation(), body, body.begin());
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

struct AllReduceOpV1ToV2 : public OpRewritePattern<AllReduceOpV1> {
  using OpRewritePattern<AllReduceOpV1>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllReduceOpV1 op,
                                PatternRewriter& rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<AllReduceOpV2>(
        op, op->getResultTypes(), op->getOperands(), op.getReplicaGroups(),
        op.getChannelId(), op.getUseGlobalDeviceIds());
    Region& body = newOp.getComputation();
    rewriter.inlineRegionBefore(op.getComputation(), body, body.begin());
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

struct AllReduceOpV2ToV1 : public OpRewritePattern<AllReduceOpV2> {
  using OpRewritePattern<AllReduceOpV2>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllReduceOpV2 op,
                                PatternRewriter& rewriter) const override {
    if (op->getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(op, "multiple operands");
    }
    auto newOp = rewriter.replaceOpWithNewOp<AllReduceOpV1>(
        op, op.getResultTypes().front(), op->getOperands().front(),
        op.getReplicaGroups(), op.getChannelId(), op.getUseGlobalDeviceIds());
    Region& body = newOp.getComputation();
    rewriter.inlineRegionBefore(op.getComputation(), body, body.begin());
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

struct CompositeOpV1ToV2 : public OpRewritePattern<CompositeOpV1> {
  using OpRewritePattern<CompositeOpV1>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompositeOpV1 op,
                                PatternRewriter& rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<CompositeOpV2>(
        op, op->getResultTypes(), op.getInputs(), op.getName(),
        op.getCompositeAttributes(), op.getDecomposition(), op.getVersion(),
        /*composite_regionsCount=*/0);
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

struct CompositeOpV2ToV1 : public OpRewritePattern<CompositeOpV2> {
  using OpRewritePattern<CompositeOpV2>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompositeOpV2 op,
                                PatternRewriter& rewriter) const override {
    if (!op.getCompositeRegions().empty()) {
      return rewriter.notifyMatchFailure(op, "non-empty regions");
    }
    auto newOp = rewriter.replaceOpWithNewOp<CompositeOpV1>(
        op, op->getResultTypes(), op.getInputs(), op.getName(),
        op.getCompositeAttributes(), op.getDecomposition(), op.getVersion());
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

struct CustomCallOpV1ToV2 : public OpRewritePattern<CustomCallOpV1> {
  CustomCallOpV1ToV2(MLIRContext* context)
      : OpRewritePattern<CustomCallOpV1>(context) {}

  LogicalResult matchAndRewrite(CustomCallOpV1 op,
                                PatternRewriter& rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<CustomCallOpV2>(
        op, op->getResultTypes(), op.getInputs(), op.getCallTargetName(),
        op.getHasSideEffect(), op.getBackendConfig(), op.getApiVersion(),
        op.getCalledComputations(), op.getOperandLayouts(),
        op.getResultLayouts(), op.getOutputOperandAliases(),
        getEmptyArray(rewriter));
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

struct CustomCallOpV2ToV1 : public OpRewritePattern<CustomCallOpV2> {
  using OpRewritePattern<CustomCallOpV2>::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomCallOpV2 op,
                                PatternRewriter& rewriter) const override {
    if (!isEmptyArray(op.getResultTilings())) {
      return rewriter.notifyMatchFailure(op, "non-empty result_tilings");
    }
    auto newOp = rewriter.replaceOpWithNewOp<CustomCallOpV1>(
        op, op->getResultTypes(), op.getInputs(), op.getCallTargetName(),
        op.getHasSideEffect(), op.getBackendConfig(), op.getApiVersion(),
        op.getCalledComputations(), op.getOperandLayouts(),
        op.getResultLayouts(), op.getOutputOperandAliases());
    copyDiscardableAttrs(op, newOp);
    return success();
  }
};

#include "stablehlo/transforms/VhloToVersionPatterns.h.inc"

}  // namespace
}  // namespace vhlo

namespace stablehlo {
void populateVhloToVersionPatterns(MLIRContext* context,
                                   RewritePatternSet* patterns,
                                   vhlo::VhloTypeConverter* converter) {
  vhlo::populateWithGenerated(*patterns);
  patterns->add<vhlo::ScatterOpV1ToV2, vhlo::ScatterOpV2ToV1>(context);
  patterns->add<vhlo::AllReduceOpV1ToV2, vhlo::AllReduceOpV2ToV1>(context);
  patterns->add<vhlo::CompositeOpV1ToV2, vhlo::CompositeOpV2ToV1>(context);
  patterns->add<vhlo::CustomCallOpV1ToV2, vhlo::CustomCallOpV2ToV1>(context);
}

}  // namespace stablehlo
}  // namespace mlir
