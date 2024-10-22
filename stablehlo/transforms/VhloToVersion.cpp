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
#include <utility>

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
  if (auto arrAttr = dyn_cast<ArrayV1Attr>(attr))
    return success(llvm::all_of(arrAttr.getValue(), [&](Attribute ele) {
      return succeeded(isLegalAttribute(ele, targetVersion));
    }));
  if (auto arrAttr = dyn_cast<DictionaryV1Attr>(attr)) {
    return success(llvm::all_of(
        arrAttr.getValue(), [&](std::pair<Attribute, Attribute> entry) {
          return succeeded(isLegalAttribute(entry.first, targetVersion)) &&
                 succeeded(isLegalAttribute(entry.second, targetVersion));
        }));
  }
  if (auto floatAttr = dyn_cast<FloatV1Attr>(attr))
    return isLegalType(floatAttr.getType(), targetVersion);
  if (auto intAttr = dyn_cast<IntegerV1Attr>(attr))
    return isLegalType(intAttr.getType(), targetVersion);
  if (auto tensorAttr = dyn_cast<TensorV1Attr>(attr))
    return isLegalType(tensorAttr.getType(), targetVersion);
  if (auto typeAttr = dyn_cast<TypeV1Attr>(attr))
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
  if (auto complex = dyn_cast<ComplexV1Type>(type))
    return isLegalType(complex.getElementType(), targetVersion);
  if (auto func = dyn_cast<FunctionV1Type>(type)) {
    auto validateType = [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion));
    };
    return success(llvm::all_of(func.getInputs(), validateType) &&
                   llvm::all_of(func.getOutputs(), validateType));
  }
  if (auto ranked = dyn_cast<RankedTensorV1Type>(type)) {
    auto encoding = ranked.getEncoding();
    if (encoding && failed(isLegalAttribute(encoding, targetVersion)))
      return failure();
    return isLegalType(ranked.getElementType(), targetVersion);
  }
  if (auto tuple = dyn_cast<TupleV1Type>(type))
    return success(llvm::all_of(tuple.getTypes(), [&](Type ele) {
      return succeeded(isLegalType(ele, targetVersion));
    }));
  if (auto quant = dyn_cast<UniformQuantizedV1Type>(type))
    return success(
        succeeded(isLegalType(quant.getStorageType(), targetVersion)) &&
        succeeded(isLegalType(quant.getExpressedType(), targetVersion)));
  if (auto unranked = dyn_cast<UnrankedTensorV1Type>(type))
    return isLegalType(unranked.getElementType(), targetVersion);

  // Is VHLO and valid version, success.
  return success();
}

bool isLegalOperation(Operation* op, const Version& targetVersion) {
  // Validate op
  auto opInterface = dyn_cast<VersionedOpInterface>(op);
  if (!opInterface) return false;
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

  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    stablehlo::populateVhloToVersionPatterns(&patterns_, &converter, context);
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
    target.addDynamicallyLegalDialect<VhloDialect>(
        [&targetVersion](Operation* op) {
          return isLegalOperation(op, targetVersion);
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

TensorV1Attr getEmptyI64Tensor(OpBuilder& builder) {
  auto shape = vhlo::RankedTensorV1Type::get(
      builder.getContext(), {0},
      vhlo::IntegerSI64V1Type::get(builder.getContext()), {});
  return vhlo::TensorV1Attr::get(builder.getContext(), shape, {});
}

bool isEmptyTensor(Attribute attr) {
  auto tensor = dyn_cast<TensorV1Attr>(attr);
  if (tensor) return tensor.getData().empty();
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
      denseElements.getRawData());
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
    return success();
  }
};

#include "stablehlo/transforms/VhloToVersionPatterns.h.inc"

}  // namespace
}  // namespace vhlo

namespace stablehlo {
void populateVhloToVersionPatterns(RewritePatternSet* patterns,
                                   TypeConverter* converter,
                                   MLIRContext* context) {
  vhlo::populateWithGenerated(*patterns);
  patterns->add<vhlo::ScatterOpV1ToV2, vhlo::ScatterOpV2ToV1>(context);
  patterns->add<vhlo::AllReduceOpV1ToV2, vhlo::AllReduceOpV2ToV1>(context);
}

}  // namespace stablehlo
}  // namespace mlir
