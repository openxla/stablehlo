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

#include "stablehlo/dialect/Serialization.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

LogicalResult serializePortableArtifact(ModuleOp module,
                                        StringRef targetVersion,
                                        raw_ostream& os) {
  MLIRContext* context = module.getContext();

  // Convert StableHLO --> VHLO. Will fail if entire program is not StableHLO.
  {
    PassManager pm(context);
    pm.addPass(stablehlo::createStablehloLegalizeToVhloPass());
    if (!succeeded(pm.run(module))) {
      return failure();
    }
  }

  // Convert VHLO --> VHLO(version x.y.z).
  // Doing separately for now since we need to improve error messaging around
  // target version failures.
  {
    PassManager pm(context);
    pm.addPass(stablehlo::createVhloToVersionPass({targetVersion.str()}));
    if (!succeeded(pm.run(module))) {
      return failure();
    }
  }

  // TODO(#1508): Consider adding a header to identify StableHLO portable
  // artifact versions.
  BytecodeWriterConfig writerConfig;
  // bytecodeVersion = 1 is what has been predominantly used in practice to
  // serialize portable StableHLO artifacts.
  // Theoretically speaking, StableHLO v0.9.0 which introduced compatibility
  // guarantees was released on 3/2/2023 and bytecodeVersion = 1 was released
  // on 3/10/2023, so there was a time period when we guaranteed compatibility
  // for StableHLO consumers which only supported bytecodeVersion = 0.
  // However, this time period (1 month of forward compatibility) has expired,
  // so it's fine to hardcode bytecodeVersion = 1 here.
  writerConfig.setDesiredBytecodeVersion(1);
  return writeBytecodeToFile(module, os, writerConfig);
}

OwningOpRef<ModuleOp> deserializePortableArtifact(StringRef sourceStr,
                                                  MLIRContext* context) {
  context->loadDialect<vhlo::VhloDialect>();
  auto module = parseSourceString<ModuleOp>(sourceStr, context);
  if (!module) {
    return nullptr;
  }

  // Convert VHLO --> VHLO(current) --> StableHLO
  PassManager pm(context);
  pm.addPass(stablehlo::createVhloToVersionPass({"current"}));
  pm.addPass(stablehlo::createVhloLegalizeToStablehloPass());
  if (!succeeded(pm.run(*module))) {
    return nullptr;
  }

  return module;
}

}  // namespace stablehlo
}  // namespace mlir
