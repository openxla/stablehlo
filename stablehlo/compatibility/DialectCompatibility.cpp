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
#include <cstdint>

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilties.h"
#include "mlir/Bytecode/BytecodeWriter.h"

#define DEBUG_TYPE "hlo-compatibility"

namespace mlir {
namespace compatibility {

FailureOr<Attribute> CompatibilityDialectInterface::applyConversion(
    Operation *op, Attribute const &version,
    llvm::StringMap<llvm::SmallVector<OpConversionAttributePair>> &map,
    std::function<bool(Attribute, Attribute)> const &comparisonFn) {
  // Find if any conversions for this given op are registered.
  OperationName mnemonic = op->getName();
  auto it = map.find(mnemonic.stripDialect());

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
  llvm::SmallVector<OpConversionAttributePair> &conversions = it->second;
  std::sort(conversions.begin(), conversions.end(),
            [&](OpConversionAttributePair const &a,
                OpConversionAttributePair const &b) {
              return comparisonFn(a.version, b.version);
            });

  // Iterate over conversions, if one is greater than version argument, apply
  // it and modify version.
  for (auto &convPair : conversions) {
    LLVM_DEBUG(llvm::dbgs() << "Trying to apply v" << convPair.version << " to "
                            << op->getName().getStringRef() << '\n');
    if (comparisonFn(version, convPair.version)) {
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
    Operation *topLevelOp,
    std::function<Attribute(Operation *)> const &getOpVersionFn,
    std::function<FailureOr<Attribute>(Operation *, Attribute const &,
                                       CompatibilityDialectInterface *)> const
        &cb) {
  // Perform any upgrades
  auto walkRes = topLevelOp->walk([&](Operation *op) {
    LLVM_DEBUG(llvm::dbgs() << "Getting dialect version for "
                            << op->getName().getStringRef() << '\n');
    Attribute version = getOpVersionFn(op);
    if (!version) return WalkResult::advance();

    FailureOr<Attribute> attrOrFail(version);
    bool appliedConversion = false;

    do {
      // Upgrade failed, interrupt and error.
      // Get this every time in case upgrade changes op dialect.
      LLVM_DEBUG(llvm::dbgs() << "Checking conversions for "
                              << op->getName().getStringRef() << '\n');
      auto *dialect = op->getDialect();
      if (!dialect) return WalkResult::advance();

      auto *interface =
          dialect->getRegisteredInterface<CompatibilityDialectInterface>();
      if (!interface)
        return WalkResult::advance();

      LLVM_DEBUG(llvm::dbgs() << "Attempting to apply conversion to "
                              << op->getName().getStringRef() << '\n');
      // Iteratively apply upgrades until none are applied.
      version = *attrOrFail;  // Update the minimum version.
      attrOrFail = cb(op, version, interface);

      if (failed(attrOrFail)) {
        // Upgrade failed, interrupt and error.
        LLVM_DEBUG(llvm::dbgs() << "Op failed to apply conversion.\n");
        return WalkResult::interrupt();
      }

      appliedConversion = (*attrOrFail != version); // FIXME: Update this.
      LLVM_DEBUG(llvm::dbgs() << "Applied conversion = "
                              << appliedConversion << '\n');

    } while (appliedConversion);
    return WalkResult::advance();
  });

  return success(/*isSuccess=*/!walkRes.wasInterrupted());
}

Attribute getOpVersionFromDialectProducerVersion(Operation *op) {
  auto *dialect = op->getDialect();
  if (!dialect) return Attribute();

  auto *interface =
      dialect->getRegisteredInterface<CompatibilityDialectInterface>();
  if (!interface) return Attribute();

  return interface->getProducerVersion();
}

Attribute getOpVersionFromDialectMap(
    llvm::StringMap<Attribute> const &dialectVersions, Operation *op) {
  auto * dialect = op->getDialect();
  if (dialect && dialectVersions.count(dialect->getNamespace())) {
    return dialectVersions.lookup(dialect->getNamespace());
  }
  return getOpVersionFromDialectProducerVersion(op);
}
}  // namespace

LogicalResult CompatibilityDialectInterface::applyOpUpgrades(
    Operation *topLevelOp, llvm::StringMap<Attribute> const &dialectVersions) {
 // FIXME: This version number would need to come from the dialect_resources:
 // FIXME: cleanup
 auto getDialectVersion = [&dialectVersions](Operation *op) {
  return getOpVersionFromDialectMap(dialectVersions, op);
 };
 return walkAndApply(topLevelOp, getDialectVersion,
                     [](Operation *op, Attribute const &attr,
                        CompatibilityDialectInterface *converter) {
                       return converter->upgrade(op, attr);
                     });
}

LogicalResult CompatibilityDialectInterface::applyOpDowngrades(
    Operation *topLevelOp, llvm::StringMap<Attribute> const &/*targetVersions*/) {
 // Version function should return the producer version of the dialect.
 return walkAndApply(topLevelOp, getOpVersionFromDialectProducerVersion,
                     [](Operation *op, Attribute const &attr,
                        CompatibilityDialectInterface *converter) {
                       return converter->downgrade(op, attr);
                     });
}

namespace {
/// Writes the target version as an attribute on the top level operation of the
/// IR.
LogicalResult writeProducerVersions(Operation *topLevelOperation) {
  llvm::SmallSet<CompatibilityDialectInterface *, 4> interfaces;
  topLevelOperation->walk([&interfaces](Operation * op) {
    auto * dialect = op->getDialect();
    if (!dialect) return WalkResult::advance();

    auto *interface =
        dialect->getRegisteredInterface<CompatibilityDialectInterface>();
    if (!interface) return WalkResult::advance();

    interfaces.insert(interface);
    return WalkResult::advance();
  });

  for (CompatibilityDialectInterface * interface : interfaces) {
    auto attrNameTwine = interface->getDialect()->getNamespace() + "_version";
    llvm::SmallVector<char> buffer;
    auto attrName = attrNameTwine.toStringRef(buffer);
    topLevelOperation->setAttr(attrName, interface->getMinimumDowngradeDialectVersion());
  }
  return success();
}

/// Checks the top level operation of the IR for a version number attribute.
/// All files produced from `writeWithCompat` must include this attribute
/// in order to provide valid compatibility guarantees.
llvm::StringMap<Attribute> extractProducerVersions(Operation *topLevelOperation) {
  llvm::StringMap<Attribute> versions;
  llvm::StringRef suffix = "_version";
  for (NamedAttribute const & attr : topLevelOperation->getAttrs()) {
    StringRef nameRef = attr.getName();
    if (nameRef.endswith(suffix)) {
      versions[nameRef.drop_back(suffix.size())] = attr.getValue();
    }
  }
  return versions;
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
  Operation *topLevelOperation = module.get();
  auto versions = extractProducerVersions(topLevelOperation);

  // Apply upgrades
  if (failed(CompatibilityDialectInterface::applyOpUpgrades(topLevelOperation,
                                                            versions))) {
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
                              llvm::StringMap<Attribute> targetVersions, bool emitBytecode,
                              llvm::raw_ostream &output) {
  if (failed(verify(topLevelOperation))) {
    return topLevelOperation->emitError("must be valid op");
  }

  // TODO: Downgrade to target version
  if (failed(CompatibilityDialectInterface::applyOpDowngrades(
          topLevelOperation, targetVersions))) {
    return failure();
  }

  // Modify top level version attribute.
  if (failed(writeProducerVersions(topLevelOperation))) {
    return failure();
  }

  // TODO: Add printer flags argument
  if (emitBytecode) {
    writeBytecodeToFile(topLevelOperation, output);
  } else {
  }
  topLevelOperation->print(output /*, printerFlags = */);
  return success();
}

}  // namespace compatibility
}  // namespace mlir
