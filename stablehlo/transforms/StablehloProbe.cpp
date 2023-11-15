/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
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
#include <string>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOPROBEPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

class StablehloProbePass
    : public impl::StablehloProbePassBase<StablehloProbePass> {
 public:
  explicit StablehloProbePass() = default;
  void runOnOperation() override;

 private:
  // Create a uniquely identifying probe ID for the given value being
  // instrumented. Attempt to use named MLIR location data, otherwise use an
  // increasing ID.
  std::string getLocationNameOrUniqueId(Location location, unsigned int& id);

  // Instrument a specified operation by adding an `interpreter.probe` op for
  // each result produced by the operation.
  void probeOp(Operation& op, unsigned int& id, OpBuilder& builder);

  // Determine if a given operation is suitable for instrumentation. A suitable
  // operation is defined as any operation which is not a ConstantOp, as such
  // ops are effectively a no-op in terms of computation.
  bool shouldProbeOp(Operation& op) const;
};

std::string StablehloProbePass::getLocationNameOrUniqueId(Location location,
                                                          unsigned int& id) {
  if (auto namedLocation = location.dyn_cast<NameLoc>())
    return namedLocation.getName().strref().split('@').first.str();

  return "probe" + std::to_string(++id);
}

void StablehloProbePass::probeOp(Operation& op, unsigned int& id,
                                 OpBuilder& builder) {
  for (Value value : op.getResults()) {
    builder.setInsertionPointAfterValue(value);
    Value instrumentedValue = builder.create<interpreter::ProbeOp>(
        value.getLoc(), value,
        StringAttr::get(&getContext(),
                        getLocationNameOrUniqueId(value.getLoc(), id)));
    value.replaceAllUsesExcept(instrumentedValue,
                               instrumentedValue.getDefiningOp());
  }
}

void StablehloProbePass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module);

  // Strictly increasing counter to uniquely identify probe operations when MLIR
  // location data is not available.
  unsigned int probeId = 0;

  module.walk([&](Operation* op) {
    if (shouldProbeOp(*op)) probeOp(*op, probeId, builder);

    return WalkResult::advance();
  });
}

bool StablehloProbePass::shouldProbeOp(Operation& op) const {
  if (isa<ConstantOp>(op)) return false;

  return true;
}

}  // namespace
}  // namespace stablehlo
}  // namespace mlir
