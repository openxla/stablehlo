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

#ifndef STABLEHLO_INTEGRATIONS_STABLEHLODIALECTCOMPATIBILITY_H
#define STABLEHLO_INTEGRATIONS_STABLEHLODIALECTCOMPATIBILITY_H

#include "mlir/IR/MLIRContext.h"
#include "stablehlo/compatibility/DialectCompatibilityBase.h"

namespace mlir {
namespace stablehlo {
/// Changelog:
///   - v0 [10/07/2022]: Base dialect compatibility state.
///
class StablehloCompatibilityConverter : public DialectCompatibilityBase {
 public:
  StablehloCompatibilityConverter(MLIRContext *context)
      : DialectCompatibilityBase(context) {
    // Add upgrades and downgrades here.
  }

  /// The current version of StableHLO serialization
  int64_t getProducerVersion() const final { return 0; }

  /// Backwards compatibility: The minimum supported version of StableHLO
  int64_t getMinimumProducerDialectVersion() const final { return 0; }

  /// Forwards compatibility: The minimum supported version of StableHLO
  int64_t getMinimumDowngradeDialectVersion() const final { return 0; }
};

/// Entrypoint for parsing a file that was serialized with compatibility
/// guarantees.
OwningOpRef<Operation *> parseWithCompat(llvm::SourceMgr &sourceMgr,
                                         MLIRContext *context) {
  StablehloCompatibilityConverter interface(context);
  return detail::parseWithCompatImpl(sourceMgr, context, interface);
}

/// Entrypoint for writing a file that was serialized with compatibility
/// guarantees.
LogicalResult writeWithCompat(Operation *topLevelOperation,
                              MLIRContext *context, int64_t targetVersion,
                              bool emitBytecode, llvm::raw_ostream &output) {
  StablehloCompatibilityConverter interface(context);
  return detail::writeWithCompatImpl(topLevelOperation, targetVersion,
                                     emitBytecode, output, interface);
}

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_INTEGRATIONS_STABLEHLODIALECTCOMPATIBILITY_H