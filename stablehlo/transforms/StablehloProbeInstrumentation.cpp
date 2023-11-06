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

#define GEN_PASS_DEF_STABLEHLOPROBEINSTRUMENTATIONPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

class StablehloProbeInstrumentationPass
    : public impl::StablehloProbeInstrumentationPassBase<
          StablehloProbeInstrumentationPass> {
 public:
  explicit StablehloProbeInstrumentationPass() = default;
  void runOnOperation() override;

 private:
  // Create a uniquely identifying probe ID for the given value being
  // instrumented. Attempt to use named MLIR location data, otherwise use an
  // increasing ID.
  std::string getLocationNameOrUniqueId(Location location);

  // Instrument a specified value.
  void instrumentValue(Value value, OpBuilder& builder);

  // Determine if a specified value should be instrumented.
  bool shouldInstrumentValue(Value value) const;

  // After all values have been instrumented, update any downstream uses of old
  // values with the new instrumentation values
  void updateAffectedOperations();

  // Map of values which have been instrumented.
  DenseMap<Value, Value> instrumentedValues;

  // Strictly increasing counter to uniquely identify probe operations when MLIR
  // location data is not available.
  unsigned int probeId = 0;
};

std::string StablehloProbeInstrumentationPass::getLocationNameOrUniqueId(
    Location location) {
  if (auto namedLocation = location.dyn_cast<NameLoc>())
    return namedLocation.getName().strref().split('@').first.str();

  return std::to_string(++probeId);
}

void StablehloProbeInstrumentationPass::instrumentValue(Value value,
                                                        OpBuilder& builder) {
  builder.setInsertionPointAfterValue(value);
  instrumentedValues[value] = builder.create<interpreter::ProbeOp>(
      value.getLoc(), value,
      StringAttr::get(&getContext(),
                      getLocationNameOrUniqueId(value.getLoc())));
}

void StablehloProbeInstrumentationPass::runOnOperation() {
  ModuleOp module = getOperation();

  OpBuilder builder(module);
  SmallVector<Value> valuesToInstrument;

  module.walk([&](Operation* op) {
    for (Value result : op->getResults()) valuesToInstrument.push_back(result);
  });

  for (Value value : valuesToInstrument)
    if (shouldInstrumentValue(value)) instrumentValue(value, builder);

  updateAffectedOperations();
}

bool StablehloProbeInstrumentationPass::shouldInstrumentValue(
    Value value) const {
  if (Operation* op = value.getDefiningOp())
    if (isa<ConstantOp>(op)) return false;

  return true;
}

void StablehloProbeInstrumentationPass::updateAffectedOperations() {
  for (auto& [value, instrumentedValue] : instrumentedValues) {
    value.replaceAllUsesExcept(instrumentedValue, instrumentedValue.getDefiningOp());
  }
}

}  // namespace
}  // namespace stablehlo
}  // namespace mlir
