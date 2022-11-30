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

set(LLVM_TARGET_DEFINITIONS VhloOps.td)
mlir_tablegen(VhloOps.h.inc -gen-op-decls)
mlir_tablegen(VhloOps.cpp.inc -gen-op-defs)
mlir_tablegen(VhloOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(VhloOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(VhloEnums.h.inc -gen-enum-decls)
mlir_tablegen(VhloEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(VhloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(VhloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(VhloOpsIncGen)
add_dependencies(mlir-headers VhloOpsIncGen)

add_mlir_dialect_library(VhloOps
  PARTIAL_SOURCES_INTENDED
  VhloOps.cpp
  # VhloBytecode.cpp

  DEPENDS
  VhloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloAssemblyFormat
  MLIRIR
  MLIRSupport
  MLIRQuantDialect
)

add_mlir_dialect_library(VhloRegister
  PARTIAL_SOURCES_INTENDED
  Register.cpp

  LINK_LIBS PUBLIC
  ChloOps
  VhloOps
)

