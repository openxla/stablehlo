/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "stablehlo/api/PortableApi.h"

#include <string>

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mlir {
namespace stablehlo {
namespace {
void loadSerializationDialects(MLIRContext* context) {
  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::stablehlo::StablehloDialect>();
  context->loadDialect<mlir::vhlo::VhloDialect>();
}
}  // namespace

std::string getCurrentVersion() {
  return mlir::vhlo::Version::getCurrentVersion().toString();
}

std::string getMinimumVersion() {
  return mlir::vhlo::Version::getMinimumVersion().toString();
}

LogicalResult serializePortableArtifact(StringRef moduleStr,
                                        StringRef targetVersion,
                                        raw_ostream& os) {
  MLIRContext context;
  loadSerializationDialects(&context);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
  if (!module || failed(module->verifyInvariants())) return failure();

  return serializePortableArtifact(*module, targetVersion, os);
}

LogicalResult deserializePortableArtifact(StringRef artifactStr,
                                          raw_ostream& os) {
  MLIRContext context;
  loadSerializationDialects(&context);
  auto module = deserializePortableArtifact(artifactStr, &context);
  if (!module) return failure();

  // This bytecode does not need to specify version number or producer string,
  // since it is not required to be any more stable than textual assembly.
  return writeBytecodeToFile(*module, os);
}

}  // namespace stablehlo
}  // namespace mlir
