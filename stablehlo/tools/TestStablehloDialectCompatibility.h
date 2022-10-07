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

#ifndef STABLEHLO_INTEGRATIONS_TESTSTABLEHLODIALECTCOMPATIBILITY_H
#define STABLEHLO_INTEGRATIONS_TESTSTABLEHLODIALECTCOMPATIBILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/compatibility/DialectCompatibilityInterface.h"

namespace mlir {
namespace stablehlo {

class TestStablehloCompatibilityConverter : public DialectCompatibilityInterface {
 public:
  TestStablehloCompatibilityConverter(MLIRContext *context)
      : DialectCompatibilityInterface(context) {
    // Add upgrades and downgrades here.
    registerAddOpChanges();
    registerSubOpChanges();
  }

  /// The current version of StableHLO serialization
  int64_t getProducerVersion() const final { return 40; }

  /// Backwards compatibility: The minimum supported version of StableHLO
  int64_t getMinimumProducerDialectVersion() const final { return 35; }

  /// Forwards compatibility: The minimum supported version of StableHLO
  int64_t getMinimumDowngradeDialectVersion() const final { return 38; }

  //==----------
  // Conversions
  //==----------
  void registerSubOpChanges() {
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

  void registerAddOpChanges() {
    // Change log:
    //   Version 37: Add has no attributes
    //   Version 38: Added attr version_38_attr
    //   Version 39: Rename attr version_38_attr --> version_39_attr
    // Backward compatibility: Support v38 and after.
    // Forward compatibility: Target v39 for printing.

    // Upgrade v38 -> v39: [version_38_attr --> version_39_attr]
    addUpgrade(
        "stablehlo.add", 39,
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

  //==------
  // Helpers
  //==------
  // FIXME: Is there a good way to change an op name?
  // This function currently doesn't handle regions.
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
  }
};

}  // namespace stablehlo
}  // namespace mlir

#endif // STABLEHLO_INTEGRATIONS_TESTSTABLEHLODIALECTCOMPATIBILITY_H