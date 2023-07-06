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

#include <cstdint>

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Version.h"
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

  // Write bytecode with producer string "StableHLO_vX.Y.Z"
  // using the bytecode format version associated with the StableHLO release.
  auto producer = "StableHLO_v" + targetVersion.str();
  BytecodeWriterConfig writerConfig(producer);
  auto bytecodeVersion =
      vhlo::Version::fromString(targetVersion)->getBytecodeVersion();
  if (failed(bytecodeVersion)) return failure();
  writerConfig.setDesiredBytecodeVersion(bytecodeVersion.value());
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
  createStablehloDeserializePipeline(pm);
  if (!succeeded(pm.run(*module))) {
    return nullptr;
  }

  return module;
}

}  // namespace stablehlo
}  // namespace mlir
