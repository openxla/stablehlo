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

#ifndef STABLEHLO_INTEGRATIONS_DIALECTCOMPATIBILITYBASE_H
#define STABLEHLO_INTEGRATIONS_DIALECTCOMPATIBILITYBASE_H

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
/// Downgrades applied after verification, before serialization.
/// Upgrades applied after deserialization, before verification.
///
/// This allows for the following types of changes:
///  - Operation/Attribute renames
///  - Move an operation to a different dialect
///  - Adding an attribute (assuming default value for upgrade is possible)
///  - Removing an attribute from an op (can't delete attribute datatype)
///  - Changing attribute into an operand / adding an operand (need to think
///  more)
class DialectCompatibilityBase {
 public:
  /// Version converter functions are used for both upgrade and downgrade.
  /// They take an `Operation *` for the operation to be conveted, as well as
  /// an int64_t that represents the current version of the op.
  using OpVersionConverterFn =
      std::function<LogicalResult(Operation *, int64_t)>;

  DialectCompatibilityBase(MLIRContext *context)
      : context(context), upgrades(), downgrades() {}

  virtual ~DialectCompatibilityBase() = default;

  /// This is the current dialect bytecode version.
  ///
  /// A warning will be displayed if the bytecode version greater than producer
  /// version. Note: Future producers will target versions less than or equal to
  /// the current producer version for the duration of the forward compatibility
  /// window.
  virtual int64_t getProducerVersion() const = 0;

  /// Backward compatibility: Returns the current minimum supported producer
  /// version.
  ///
  /// A warning will be displayed if the bytecode version is less than minimum
  /// supported version.
  virtual int64_t getMinimumProducerDialectVersion() const = 0;

  /// Forward compatibility: Returns the downgrade target version.
  /// It should be set to the `getProducerDialectVersion` of a revision
  /// from <forward_compatibility_window> days in the past.
  ///
  /// This is the default target version for stablehlo-translate.
  virtual int64_t getMinimumDowngradeDialectVersion() const = 0;

 protected:
  /// Add an upgrade/downgrade pass for an Operation that matches @param
  /// mnemonic.
  ///
  /// Given that operations being deserialized here may no longer exist, this
  /// machinery operates on mnemonic strings and passes an `Operation*` for
  /// upgrade.
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

  MLIRContext *getContext() { return context; }
  MLIRContext const *getContext() const { return context; }

 public:
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
  LogicalResult applyOpUpgrades(Operation *topLevelOp,
                                int64_t const &fileVersion);

  LogicalResult applyOpDowngrades(Operation *topLevelOp,
                                  int64_t const &targetVersion);

 private:
  struct OpConversionVersionPair {
    OpVersionConverterFn conversion;
    int64_t version;
  };

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

  /// Call applyConversion until no changes made, or targetVersion reached.
  FailureOr<int64_t> applyConversions(
      Operation *op, int64_t version, int64_t const targetVersion,
      llvm::StringMap<llvm::SmallVector<OpConversionVersionPair>> &map,
      std::function<bool(int64_t, int64_t)> const &comparisonFn) {
    bool hasChanged = true;
    while (hasChanged) {
      auto conversionResult =
          applyConversion(op, version, targetVersion, map, comparisonFn);
      if (failed(conversionResult)) {
        return failure();
      }
      hasChanged = (version != *conversionResult);
      version = *conversionResult;
    }
    return version;
  }

  FailureOr<int64_t> upgrade(Operation *op, int64_t version) {
    return applyConversions(op, version, getProducerVersion(), upgrades,
                            std::less<int64_t>());
  }

  FailureOr<int64_t> downgrade(Operation *op, int64_t version,
                               int64_t targetVersion) {
    return applyConversions(op, version, targetVersion, downgrades,
                            std::greater<int64_t>());
  }

  MLIRContext *context;
  llvm::StringMap<llvm::SmallVector<OpConversionVersionPair>> upgrades;
  llvm::StringMap<llvm::SmallVector<OpConversionVersionPair>> downgrades;
};

struct CompatOptions {
  int64_t targetVersion = -1;
  bool emitAssembly = false;
};

namespace detail {
/// Separate impl funciton for testing.
OwningOpRef<Operation *> parseWithCompatImpl(
    llvm::SourceMgr &sourceMgr, MLIRContext *context,
    DialectCompatibilityBase &interface);

/// Separate impl funciton for testing.
LogicalResult writeWithCompatImpl(Operation *topLevelOperation,
                                  CompatOptions opts, llvm::raw_ostream &output,
                                  DialectCompatibilityBase &interface);
}  // namespace detail

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_INTEGRATIONS_DIALECTCOMPATIBILITYBASE_H