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

#include "stablehlo/compatibility/DialectCompatibility.h"
#include <algorithm>
#include <cstdint>
#include <memory>

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilties.h"
#include "mlir/Bytecode/BytecodeWriter.h"

#define DEBUG_TYPE "hlo-compatibility"

namespace mlir {
namespace stablehlo {

FailureOr<int64_t> StablehloCompatibilityConverter::applyConversion(
    Operation *op, int64_t const version, int64_t const targetVersion,
    llvm::StringMap<llvm::SmallVector<OpConversionVersionPair>> &map,
    std::function<bool(int64_t, int64_t)> const &comparisonFn) {
  // Find if any conversions for this given op are registered.
  OperationName mnemonic = op->getName();
  auto it = map.find(mnemonic.getStringRef());

  // No conversions, return original version.
  if (it == map.end()) {
    return version;
  }

  // Sort the conversions for this given attribute and see if any can be
  // applied.
  // This will either sort in ascending or descending order depending on
  // comparisonFn.
  //
  // Sort on every conversion may be costly, can consider refactoring.
  llvm::SmallVector<OpConversionVersionPair> &conversions = it->second;
  std::sort(
      conversions.begin(), conversions.end(),
      [&](OpConversionVersionPair const &a, OpConversionVersionPair const &b) {
        return comparisonFn(a.version, b.version);
      });

  // Iterate over conversions, if one is greater than version argument, apply
  // it and modify version.
  for (auto &convPair : conversions) {
    // Apply if comparison fn returns true, and conversion funciton is
    // a valid up/downgrade with the given target version.
    bool shouldApply = comparisonFn(version, convPair.version) &&
                       (comparisonFn(convPair.version, targetVersion) ||
                        convPair.version == targetVersion);
    if (shouldApply) {
      // If conversion was attempted, return failure or new attribute version.
      if (failed(convPair.conversion(op, version))) {
        LLVM_DEBUG(llvm::dbgs() << "Op failed to apply conversion "
                                << convPair.version << '\n');
        return failure();
      }
      assert(convPair.version != version);  // Must always increase/decrease.
      return convPair.version;              // Return the new version
    }
  }

  // No conversions applied, return original version.
  return version;
}

namespace {
LogicalResult walkAndApply(
    Operation *topLevelOp, int64_t version,
    std::function<FailureOr<int64_t>(Operation *)> const &cb) {
  // Perform any upgrades
  auto walkRes = topLevelOp->walk([&](Operation *op) {
    auto newVersion = cb(op);
    if (failed(newVersion)) {
      // Upgrade failed, interrupt and error.
      LLVM_DEBUG(llvm::dbgs() << "Op failed to apply conversion.\n");
      return WalkResult::interrupt();
    }

    LLVM_DEBUG(llvm::dbgs() << "Converted op v"
                            << version << " -> v" << *newVersion << " ("
                            << op->getName().getStringRef() << ")\n");

    return WalkResult::advance();
  });

  return success(/*isSuccess=*/!walkRes.wasInterrupted());
}
}  // namespace

LogicalResult StablehloCompatibilityConverter::applyOpUpgrades(
    Operation *topLevelOp, int64_t const &fileVersion) {
  return walkAndApply(topLevelOp, fileVersion,
                      [&](Operation *op) { return upgrade(op, fileVersion); });
}

LogicalResult StablehloCompatibilityConverter::applyOpDowngrades(
    Operation *topLevelOp, int64_t const & targetVersion) {
  return walkAndApply(topLevelOp, getProducerVersion(), [&](Operation *op) {
    return downgrade(op, getProducerVersion(), targetVersion);
  });
}

namespace {
/// Writes the target version as an attribute on the top level operation of the
/// IR.
LogicalResult writeProducerVersion(Operation *topLevelOperation,
                                   int64_t const &version) {
  auto attrName = "compat_version";
  topLevelOperation->setAttr(
      attrName,
      Builder(topLevelOperation->getContext()).getI64IntegerAttr(version));
  return success();
}

/// Checks the top level operation of the IR for a version number attribute.
/// All files produced from `writeWithCompat` must include this attribute
/// in order to provide valid compatibility guarantees.
FailureOr<int64_t> extractProducerVersion(Operation *topLevelOperation) {
  llvm::StringRef attrName = "compat_version";
  if (!topLevelOperation->hasAttr(attrName)) {
    return failure();
  }
  auto versionAttr =
      topLevelOperation->getAttr(attrName).dyn_cast<IntegerAttr>();
  if (!versionAttr) {
    return topLevelOperation->emitError("expected integer version");
  }
  return versionAttr.getInt();
}

}  // namespace

OwningOpRef<Operation *> parseWithCompat(llvm::SourceMgr &sourceMgr,
                                         MLIRContext *context) {
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig config(context, /*verifyAfterParse=*/false,
                      &fallbackResourceMap);
  OwningOpRef<Operation *> module =
      parseSourceFileForTool(sourceMgr, config, /*implicitModule=*/false);

  // Check for parse errors.
  if (!module) {
    return nullptr;
  }

  // Check that top level op has a valid version number.
  StablehloCompatibilityConverter converter(context);
  Operation *topLevelOperation = module.get();
  auto version = extractProducerVersion(topLevelOperation);
  if (failed(version)) {
    version = converter.getProducerVersion();
  }

  // Check that file is supported by current libStablehlo
  if (version > converter.getProducerVersion()) {
    topLevelOperation->emitWarning()
        << "file version " << *version
        << " is greater than the StableHLO consumer version "
        << converter.getProducerVersion()
        << ". Compatibility is not guaranteed.";
  }
  if (version < converter.getMinimumProducerDialectVersion()) {
    topLevelOperation->emitWarning()
        << "file version " << *version
        << " is less than the minimum suported StableHLO file version "
        << converter.getMinimumProducerDialectVersion()
        << ". Compatibility is not guaranteed.";
  }

  //  Apply upgrades
  if (failed(converter.applyOpUpgrades(topLevelOperation, *version))) {
    topLevelOperation->emitError("failed to apply upgrade");
    return nullptr;
  }

  // Verify that op is valid after upgrades
  if (failed(verify(topLevelOperation))) {
    topLevelOperation->emitError("failed to verify");
    return nullptr;
  }

  return module.release();
}

LogicalResult writeWithCompat(Operation *topLevelOperation,
                              int64_t const &targetVersion, bool emitBytecode,
                              llvm::raw_ostream &output) {
  if (failed(verify(topLevelOperation))) {
    return topLevelOperation->emitError("must be valid op");
  }

  // TODO: Downgrade to target version
  StablehloCompatibilityConverter converter(topLevelOperation->getContext());
  int64_t producerVersion = std::min(targetVersion, converter.getProducerVersion());
  producerVersion = std::max(targetVersion, converter.getMinimumDowngradeDialectVersion());
  if (failed(converter.applyOpDowngrades(topLevelOperation, targetVersion))) {
    return failure();
  }

  // Modify top level version attribute.
  // TODO: Add validation to the target version number.
  if (failed(writeProducerVersion(topLevelOperation, producerVersion))) {
    return failure();
  }

  // Emit IR or bytecode
  if (emitBytecode) {
    writeBytecodeToFile(topLevelOperation, output);
  } else {
    topLevelOperation->print(output /*, printerFlags = */);
  }
  return success();
}

namespace {

// FIXME: I'm sure there is a better way to do this.
// Rename input op to new name.
void renameOperation(Operation *op, llvm::StringRef newName) {
  class SimpleRewriter : public PatternRewriter {
   public:
    SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
  };
  OperationState state(op->getLoc(), newName);
  state.addAttributes(op->getAttrs());
  state.addOperands(op->getOperands());
  state.addTypes(op->getResultTypes());
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  Operation &newOp = *builder.create(state);
  SimpleRewriter rewriter(op->getContext());
  rewriter.replaceOp(op, {newOp.getResult(0)});
};

}

// FIXME: ANCHOR comment for compiler brown bag.

// ===-------------------------------------------------------------------------
// StableHLO Compatibility Code - This should evolve with the dialect.
// ===-------------------------------------------------------------------------

/// The current version of StableHLO
int64_t StablehloCompatibilityConverter::getProducerVersion() const {
  return 40;
}

/// Backwards compatibility: The minimum supported version of StableHLO
int64_t StablehloCompatibilityConverter::getMinimumProducerDialectVersion() const {
  return 35;
}

/// Forwards compatibility: The minimum supported version of StableHLO
int64_t StablehloCompatibilityConverter::getMinimumDowngradeDialectVersion() const {
  return 38;
}

void StablehloCompatibilityConverter::registerSubOpChanges() {
  // Change log:
  //   Version 39: SubOp<"stablehlo.sub"> exists
  //   Version 40: SubOp<"stablehlo.sub"> -> SubtractOp<"stablehlo.subtract">
  // Backward compatibility: Support v39 and after.
  // Forward compatibility: Target v39 for printing.

  // Upgrade <v40 -> 40: [sub --> subtract]
  addUpgrade("stablehlo.sub", 40,
             [&](Operation *op, int64_t fromVer) -> LogicalResult {
               renameOperation(op, "stablehlo.subtract");
               return success();
             });

  // Downgrade v39 -> 38: [subtract --> sub]
  addDowngrade("stablehlo.subtract", 39,
               [&](Operation *op, int64_t fromVer) -> LogicalResult {
                 renameOperation(op, "stablehlo.sub");
                 return success();
               });
}

void StablehloCompatibilityConverter::registerAddOpChanges() {
  // Change log:
  //   Version 37: Add has no attributes
  //   Version 38: Added attr version_38_attr
  //   Version 39: Rename attr version_38_attr --> version_39_attr
  // Backward compatibility: Support v38 and after.
  // Forward compatibility: Target v39 for printing.

  // Upgrade v38 -> v39: [version_38_attr --> version_39_attr]
  addUpgrade("stablehlo.add", 39,
             [&](Operation *op, int64_t fromVer) -> LogicalResult {
               if (!op->hasAttr("version_38_attr")) {
                 return op->emitError("expected version_38_attr for upgrade.");
               }
               op->setAttr("version_39_attr", op->getAttr("version_38_attr"));
               op->removeAttr("version_38_attr");
               return success();
             });

  // Upgrade <v38 -> 38: [() --> version_38_attr]
  addUpgrade("stablehlo.add", 38, [&](Operation *op, int64_t) {
    op->setAttr("version_38_attr",
                Builder(op->getContext()).getI64IntegerAttr(1));
    return success();
  });

  // Downgrade v39 -> v38
  addDowngrade(
      "stablehlo.add", 38,
      [&](Operation *op, int64_t fromVer) -> LogicalResult {
        if (!op->hasAttr("version_39_attr")) {
          return op->emitError("expected version_39_attr for downrade.");
        }
        op->setAttr("version_38_attr", op->getAttr("version_39_attr"));
        op->removeAttr("version_39_attr");
        return success();
      });
}

}  // namespace compatibility
}  // namespace mlir
