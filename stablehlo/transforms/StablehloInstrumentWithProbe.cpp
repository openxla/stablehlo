/* Copyright 2023 The StableHLO Authors.

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

#define GEN_PASS_DEF_STABLEHLOINSTRUMENTWITHPROBEPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

class StablehloInstrumentWithProbePass
    : public impl::StablehloInstrumentWithProbePassBase<
          StablehloInstrumentWithProbePass> {
 public:
  StablehloInstrumentWithProbePass()
      : StablehloInstrumentWithProbePassBase<
            StablehloInstrumentWithProbePass>() {}
  StablehloInstrumentWithProbePass(
      const StablehloInstrumentWithProbePassOptions& opts)
      : StablehloInstrumentWithProbePassBase<StablehloInstrumentWithProbePass>(
            opts){};
  void runOnOperation() override;

 private:
  // Create a uniquely identifying probe_id in the form `probe_id#` where
  // `probe_id` is either the MLIR location data (`NamedLoc(probe_id@...)`
  // followed by a . separator), or `probe` if debug information is not present
  // or used, and # is an increasing positive integer.
  std::string getLocationNameOrUniqueId(Location location, unsigned int id);

  // Instrument a specified operation by adding an `interpreter.probe` op for
  // each result produced by the operation.
  void probeValue(Value value, const std::string& probe_id, OpBuilder& builder);

  // Determine if a given operation is suitable for instrumentation. A suitable
  // operation is defined as any operation which is not a ConstantOp, and that
  // has at least 1 return value.
  bool shouldProbeOp(Operation& op) const;

  // Determine if a given value can be instrumented. Only values that are of
  // TensorType are suitable for instrumentation
  bool shouldProbeValue(Value value) const;
};

std::string StablehloInstrumentWithProbePass::getLocationNameOrUniqueId(
    Location location, unsigned int id) {
  auto namedLocation = location.dyn_cast<NameLoc>();
  std::string probeName = "probe";

  if (useDebugInfoOption && namedLocation)
    // Append a '.' to the end of the MLIR location data to make it easy to
    // extract the location data from the unique ID.
    probeName = namedLocation.getName().strref().split('@').first.str() + '.';

  return probeName + std::to_string(id);
}

void StablehloInstrumentWithProbePass::probeValue(Value value,
                                                  const std::string& probe_id,
                                                  OpBuilder& builder) {
  builder.setInsertionPointAfterValue(value);
  Value instrumentedValue = builder.create<interpreter::ProbeOp>(
      value.getLoc(), value, StringAttr::get(&getContext(), probe_id));
  value.replaceAllUsesExcept(instrumentedValue,
                             instrumentedValue.getDefiningOp());
}

void StablehloInstrumentWithProbePass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module);

  // Strictly increasing counter to uniquely identify probe operations when MLIR
  // location data is not available/used.
  unsigned int probeId = 0;

  module.walk([&](Operation* op) {
    if (!shouldProbeOp(*op)) return WalkResult::advance();

    for (auto res : op->getResults()) {
      if (shouldProbeValue(res))
        probeValue(res, getLocationNameOrUniqueId(op->getLoc(), ++probeId),
                   builder);
    }

    return WalkResult::advance();
  });
}

bool StablehloInstrumentWithProbePass::shouldProbeOp(Operation& op) const {
  if (isa<ConstantOp>(op)) return false;

  // Operations that do not produce values should not be instrumented (ReturnOp,
  // TraceOp, etc.)
  if (op.getNumResults() == 0) return false;

  return true;
}

bool StablehloInstrumentWithProbePass::shouldProbeValue(Value value) const {
  return value.getType().isa<TensorType>();
}

}  // namespace
}  // namespace stablehlo
}  // namespace mlir
