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

#ifndef STABLEHLO_INTEGRATIONS_DIALECTCOMPATIBILITY_H
#define STABLEHLO_INTEGRATIONS_DIALECTCOMPATIBILITY_H

#include <cstdint>
#include <functional>
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace stablehlo {

//===--------------------------------------------------------------------===//
// Versioning
//===--------------------------------------------------------------------===//

/// Class used for upgrade and downgrade hook management / appliction.
/// Optionally applied during asm printing using `--mlir-print-with-downgrades`
/// Always applied during bytecode serialization / deserialization.
///
/// This allows for the following types of changes:
///  - Operation/int64_t renames
///  - Move an operation to a different dialect
///  - Adding an attribute (assuming default value for upgrade is possible)
///  - Removing an attribute from an op (can't delete attribute datatype)
///  - Changing attribute into an operand / adding an operand (need to think more)
class DialectCompatibilityInterface {
 public:
  /// Version converter functions are used for both upgrade and downgrade.
  /// They take an `Operation *` for the operation to be conveted, as well as
  /// an int64_t that represents the dialect version.
  ///
  /// The int64_t argument may not be needed if the infrastructure handles
  /// these changes. I.e. upgrades will simply know to "upgrade from the
  /// version right before this one".
  using OpVersionConverterFn = std::function<LogicalResult(Operation *, int64_t)>;

  DialectCompatibilityInterface(MLIRContext *context)
      : context(context), upgrades(), downgrades() {}

  virtual ~DialectCompatibilityInterface() = default;

  /// This is the current dialect bytecode version.
  ///
  /// This is the number that will be passed to upgrade passes.
  ///
  /// An error/warning will be displayed if the bytecode version greater
  /// than 42. Note: Version 43 should target v42 bytecode for 3 weeks. If the
  /// producer version is greater than 42, that means the forward compatibility
  /// window is closed.
  virtual int64_t getProducerVersion() const = 0;

  /// This is the current dialect bytecode version.
  ///
  /// An error/warning will be displayed if the bytecode version is less
  /// than 35.
  virtual int64_t getMinimumProducerDialectVersion() const = 0;

  /// The target version will need to be manually managed.
  /// It should be set to the `getProducerDialectVersion` of a revision
  /// from <forward_compatibility_window> days in the past.
  ///
  /// This is the number that will be passed to downgrade passes.
  ///
  /// The attribute returned from `getMinimumDowngradeDialectVersion` will be
  /// added to the dialect resources in the bytecoded file. This is taken care
  /// of by the BytecodeDialectInterface during bytecode writing. If this method
  /// returns a null attribute, no version is written.
  ///
  /// Post-condition: `getMinimumDowngradeDialectVersion() <=
  /// getProducerDialectVersion()`. If this method is implemented,
  /// `getProducerDialectVersion` must return a non-null attribute.
  // FIXME: Revisit all these descriptions.
  virtual int64_t getMinimumDowngradeDialectVersion() const = 0;

 protected:
  /// Add an upgrade/downgrade pass for an Operation that matches @param
  /// mnemonic.
  ///
  /// Given that operations being deserialized here may no longer exist, this
  /// machinery needs to operate on mnemonic and pass an `Operation*` for
  /// upgrade.
  ///
  /// This machinery allows for handling renames, deletes, and other
  /// modifications.
  ///
  /// An upgrade callback will be invoked if the op matches the mnemonic, and
  /// the registered version is greater than the ops current version. An op will
  /// continue calling upgrades until all registered upgrades for a given
  /// mnemonic are invalid or return failure.
  ///
  /// If an upgrade succeeds, the ops "current version" is assigned to the
  /// registered version of the upgrade callback. This prevents infinite
  /// recursion, since version attributes are monotonically increasing.
  ///
  /// The inverse must be true for downgrades, calling only if version is less
  /// than the ops current version, with a monotonically decreasing version
  /// attribute.
  void addUpgrade(llvm::StringRef mnemonic, int64_t version,
                  OpVersionConverterFn const &cb) {
    if (version < getMinimumProducerDialectVersion()) {
      // Downgrade callback will never be used.
      mlir::emitError(mlir::UnknownLoc::get(getContext()))
          << "attempt to add upgrade that is less than supported dialect "
             "version: "
          << version;
      return;
    }

    upgrades[mnemonic].push_back({cb, version});
  }
  void addDowngrade(llvm::StringRef mnemonic, int64_t version,
                    OpVersionConverterFn const &cb) {
    if (version < getMinimumDowngradeDialectVersion()) {
      // Downgrade callback will never be used.
      mlir::emitError(mlir::UnknownLoc::get(getContext()))
          << "attempt to add downgrade that is less than target dialect "
             "verison: "
          << version;
      return;
    }

    downgrades[mnemonic].push_back({cb, version});
  }

  MLIRContext * getContext() {
    return context;
  }
  MLIRContext const * getContext() const {
    return context;
  }

 public:
  LogicalResult applyOpUpgrades(Operation *topLevelOp,
                                int64_t const &fileVersion);

  LogicalResult applyOpDowngrades(Operation *topLevelOp,
                                  int64_t const &targetVersion);

 private:
  struct OpConversionVersionPair {
    OpVersionConverterFn conversion;
    int64_t version;
  };

  /// Registration for each change in the change log.
  void registerAddOpChanges();
  void registerSubOpChanges();

  /// add all the upgrade / downgrade functions above.
  void registerUpgradesAndDowngrades() {
    registerAddOpChanges();
    registerSubOpChanges();
  }

  /// This function attempts to apply a single conversion to @param op.
  ///
  /// Returns `failure` iff a conversion was attempted to be applied and
  /// failed. Returns @param version if no conversaions were applied.
  /// Returns a new attribute representing the new version of an op if
  /// conversion was applied.
  FailureOr<int64_t> applyConversion(
      Operation *op, int64_t const version, int64_t const targetVersion,
      llvm::StringMap<llvm::SmallVector<OpConversionVersionPair>> &map,
      std::function<bool(int64_t, int64_t)> const &comparisonFn);

  FailureOr<int64_t> upgrade(Operation *op, int64_t version) {
    bool hasChanged = true;
    while (hasChanged) {
      auto conversionResult = applyConversion(op, version, getProducerVersion(),
                                              upgrades, std::less<int64_t>());
      if (failed(conversionResult)) {
        return failure();
      }
      hasChanged = (version != *conversionResult);
      version = *conversionResult;
    }
    return version;
  }

  FailureOr<int64_t> downgrade(Operation *op, int64_t version, int64_t targetVersion) {
    bool hasChanged = true;
    while (hasChanged) {
      if (version <= targetVersion) return version;
      auto conversionResult = applyConversion(
          op, version, targetVersion, downgrades, std::greater<int64_t>());
      if (failed(conversionResult)) {
        return failure();
      }
      hasChanged = (version != *conversionResult);
      version = *conversionResult;
    }
    return version;
  }

  MLIRContext *context;
  llvm::StringMap<llvm::SmallVector<OpConversionVersionPair>> upgrades;
  llvm::StringMap<llvm::SmallVector<OpConversionVersionPair>> downgrades;
};

OwningOpRef<Operation *> parseWithCompat(
    llvm::SourceMgr &sourceMgr, MLIRContext *context,
    DialectCompatibilityInterface &interface);

LogicalResult writeWithCompat(Operation *topLevelOperation,
                              int64_t const &targetVersion, bool emitBytecode,
                              llvm::raw_ostream &output,
                              DialectCompatibilityInterface &interface);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_INTEGRATIONS_DIALECTCOMPATIBILITY_H