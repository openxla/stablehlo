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

#include "stablehlo/compatibility/DialectCompatibilityBase.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Bytecode/BytecodeWriter.h"
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

#define DEBUG_TYPE "hlo-compatibility"

namespace mlir {
namespace stablehlo {

FailureOr<int64_t> DialectCompatibilityBase::applyConversion(
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
  llvm::SmallVector<OpConversionVersionPair> &conversions = it->second;
  std::sort(
      conversions.begin(), conversions.end(),
      [&](OpConversionVersionPair const &a, OpConversionVersionPair const &b) {
        return comparisonFn(a.version, b.version);
      });

  // Iterate over conversions, if one is greater than version argument, apply
  // it and modify version.
  for (auto &convPair : conversions) {
    // Apply downgrade if convPair is lt current version and lte targetVersion
    // Apply upgrade if convPair is gt version and gte targetVersion
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
    Operation *topLevelOp, int64_t originalVersion,
    std::function<FailureOr<int64_t>(Operation *)> const &convertFn) {
  // Perform any upgrades
  auto walkRes = topLevelOp->walk([&](Operation *op) {
    auto newVersion = convertFn(op);
    if (failed(newVersion)) {
      // Upgrade failed, interrupt and error.
      LLVM_DEBUG(llvm::dbgs() << "Op failed to apply conversion.\n");
      return WalkResult::interrupt();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Converted op v" << originalVersion << " -> v" << *newVersion
               << " (" << op->getName().getStringRef() << ")\n");

    return WalkResult::advance();
  });

  return success(/*isSuccess=*/!walkRes.wasInterrupted());
}
}  // namespace

LogicalResult DialectCompatibilityBase::applyOpUpgrades(
    Operation *topLevelOp, int64_t const &fileVersion) {
  return walkAndApply(topLevelOp, fileVersion,
                      [&](Operation *op) { return upgrade(op, fileVersion); });
}

LogicalResult DialectCompatibilityBase::applyOpDowngrades(
    Operation *topLevelOp, int64_t const &targetVersion) {
  return walkAndApply(topLevelOp, getProducerVersion(), [&](Operation *op) {
    return downgrade(op, getProducerVersion(), targetVersion);
  });
}

namespace {
/// Writes the target version as an attribute on the top level operation of the
/// IR.
LogicalResult writeProducerVersion(Operation *topLevelOperation,
                                   int64_t const &version) {
  auto attrName = "stablehlo.compat_version";
  topLevelOperation->setAttr(
      attrName,
      Builder(topLevelOperation->getContext()).getI64IntegerAttr(version));
  return success();
}

/// Checks the top level operation of the IR for a version number attribute.
/// All files produced from `writeWithCompat` must include this attribute
/// in order to provide valid compatibility guarantees.
FailureOr<int64_t> extractProducerVersion(Operation *topLevelOperation) {
  llvm::StringRef attrName = "stablehlo.compat_version";
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
                                         MLIRContext *context,
                                         DialectCompatibilityBase &interface) {
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
  Operation *topLevelOperation = module.get();
  auto version = extractProducerVersion(topLevelOperation);
  if (failed(version)) {
    version = interface.getProducerVersion();
  }

  // Check that file is supported by current libStablehlo
  if (version > interface.getProducerVersion()) {
    topLevelOperation->emitWarning()
        << "file version " << *version
        << " is greater than the StableHLO consumer version "
        << interface.getProducerVersion()
        << ". Compatibility is not guaranteed.";
  }
  if (version < interface.getMinimumProducerDialectVersion()) {
    topLevelOperation->emitWarning()
        << "file version " << *version
        << " is less than the minimum suported StableHLO file version "
        << interface.getMinimumProducerDialectVersion()
        << ". Compatibility is not guaranteed.";
  }

  //  Apply upgrades
  if (failed(interface.applyOpUpgrades(topLevelOperation, *version))) {
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
                              int64_t targetVersion, bool emitBytecode,
                              llvm::raw_ostream &output,
                              DialectCompatibilityBase &interface) {
  if (failed(verify(topLevelOperation))) {
    return topLevelOperation->emitError("must be valid op");
  }

  // TODO: Downgrade to target version
  int64_t producerVersion =
      std::min(targetVersion, interface.getProducerVersion());
  producerVersion =
      std::max(targetVersion, interface.getMinimumDowngradeDialectVersion());
  if (failed(interface.applyOpDowngrades(topLevelOperation, targetVersion))) {
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

}  // namespace stablehlo
}  // namespace mlir
