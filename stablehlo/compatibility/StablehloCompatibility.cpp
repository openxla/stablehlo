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

#include "mlir/IR/Builders.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/compatibility/DialectCompatibility.h"

namespace mlir {
namespace stablehlo {

class StablehloCompatibilityInterface : public compatibility::CompatibilityDialectInterface {
 public:
  StablehloCompatibilityInterface(Dialect *dialect)
      : compatibility::CompatibilityDialectInterface(dialect) {
    // Change log:
    //   Version 38: No attribute
    //   Version 39: Added attr version_39_attr
    //   Version 40: Rename attr version_39_attr --> version_40_attr
    // Backward compatibility: Support v38 and after.
    // Forward compatibility: Target v39 for printing.

    // Upgrade v39 -> 40: [version_39_attr --> version_40_attr]
    addUpgrade(
        "add", getVersion(40),
        [&](Operation *op, Attribute fromVer) -> LogicalResult {
          if (!op->hasAttr("version_39_attr")) {
            return op->emitError("expected version_39_attr for upgrade.");
          }
          op->setAttr("version_40_attr", op->getAttr("version_39_attr"));
          op->removeAttr("version_39_attr");
          return success();
        });

    // Upgrade <v39 -> 39: [() --> version_39_attr]
    addUpgrade("add", getVersion(39), [&](Operation *op, Attribute) {
      op->setAttr("version_39_attr", getVersion(1));
      return success();
    });

    // Downgrade v40 -> v39
    addDowngrade(
        "add", getVersion(39),
        [&](Operation *op, Attribute fromVer) -> LogicalResult {
          if (!op->hasAttr("version_40_attr")) {
            return op->emitError("expected version_40_attr for downrade.");
          }
          op->setAttr("version_39_attr", op->getAttr("version_40_attr"));
          op->removeAttr("version_40_attr");
          return success();
        });
  }

  Attribute getProducerVersion() const final { return getVersion(40); }

  Attribute getMinimumProducerDialectVersion() const final {
    return getVersion(35);
  }

  Attribute getMinimumDowngradeDialectVersion() const final {
    return getVersion(38);
  }

  bool lessThan(const Attribute &a, const Attribute &b) const final {
    return a.cast<IntegerAttr>().getInt() < b.cast<IntegerAttr>().getInt();
  }

 private:
  /// Helper function to get a version as an integer attr
  Attribute getVersion(int64_t ver) const {
    return Builder(getDialect()->getContext()).getI32IntegerAttr(ver);
  }
};

void addCompatibilityInterface(StablehloDialect *dialect)
{
  dialect->addInterface<StablehloCompatibilityInterface>();
}

}  // namespace stablehlo
}  // namespace mlir