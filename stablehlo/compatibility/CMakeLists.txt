# Copyright 2022 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(LLVM_TARGET_DEFINITIONS VersionedStablehloOps.td)
mlir_tablegen(VersionedStablehloOps.h.inc -gen-op-decls)
mlir_tablegen(VersionedStablehloOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(VersionedStablehloOpsIncGen)
add_dependencies(mlir-headers VersionedStablehloOpsIncGen)

add_mlir_dialect_library(VersionedStablehloOps
  PARTIAL_SOURCES_INTENDED
  VersionedStablehloOps.cpp
  # VersionedStablehloBytecode.cpp

  DEPENDS
  VersionedStablehloOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

# Add base compatibility library
add_mlir_library(DialectCompatibility
  PARTIAL_SOURCES_INTENDED
  DialectCompatibilityBase.cpp

  LINK_LIBS PUBLIC
  #VersionedStablehloOps
  MLIRIR
  MLIRParser
  MLIRBytecodeWriter
)
