# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2023 The StableHLO Authors.
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

set(LLVM_TARGET_DEFINITIONS Base.td)
mlir_tablegen(BaseAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(BaseAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(StablehloBaseIncGen)
add_dependencies(mlir-headers StablehloBaseIncGen)

add_mlir_library(StablehloBase
  PARTIAL_SOURCES_INTENDED
  Base.cpp

  DEPENDS
  StablehloBaseIncGen

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSupport
)

add_mlir_library(StablehloBroadcastUtils
  PARTIAL_SOURCES_INTENDED
  BroadcastUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRShapeDialect
  MLIRSupport
)

set(LLVM_TARGET_DEFINITIONS ChloOps.td)
mlir_tablegen(ChloOps.h.inc -gen-op-decls)
mlir_tablegen(ChloOps.cpp.inc -gen-op-defs)
mlir_tablegen(ChloEnums.h.inc -gen-enum-decls)
mlir_tablegen(ChloEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ChloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(ChloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(ChloOpsIncGen)
add_dependencies(mlir-headers ChloOpsIncGen)

add_mlir_dialect_library(ChloOps
  PARTIAL_SOURCES_INTENDED
  ChloOps.cpp
  ChloBytecode.cpp

  DEPENDS
  ChloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloBase
  StablehloBroadcastUtils
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRQuantDialect
  MLIRTransformUtils
)

target_include_directories(ChloOps INTERFACE
  $<BUILD_INTERFACE:${STABLEHLO_SOURCE_DIR}>
  $<BUILD_INTERFACE:${STABLEHLO_BINARY_DIR}>
)

add_mlir_dialect_library(StablehloRegister
  PARTIAL_SOURCES_INTENDED
  Register.cpp

  DEPENDS
  ChloOpsIncGen
  StablehloOpsIncGen
  VhloOpsIncGen

  LINK_LIBS PUBLIC
  ChloOps
  StablehloOps
  VhloOps
)

add_mlir_dialect_library(StablehloAssemblyFormat
  PARTIAL_SOURCES_INTENDED
  AssemblyFormat.cpp

  LINK_LIBS PUBLIC
  StablehloBase
  MLIRIR
)

add_mlir_dialect_library(StablehloSerialization
  PARTIAL_SOURCES_INTENDED
  Serialization.cpp

  LINK_LIBS PUBLIC
  StablehloOps
  StablehloPasses
  VhloOps
  MLIRBytecodeWriter
  MLIRIR
)

add_mlir_dialect_library(StablehloTypeInference
  PARTIAL_SOURCES_INTENDED
  TypeInference.cpp

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR
  StablehloAssemblyFormat
  StablehloBase
)

set(LLVM_TARGET_DEFINITIONS StablehloOps.td)
mlir_tablegen(StablehloOps.h.inc -gen-op-decls)
mlir_tablegen(StablehloOps.cpp.inc -gen-op-defs)
mlir_tablegen(StablehloEnums.h.inc -gen-enum-decls)
mlir_tablegen(StablehloEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(StablehloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(StablehloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(StablehloOpsIncGen)
add_dependencies(mlir-headers StablehloOpsIncGen)

add_mlir_dialect_library(StablehloOps
  PARTIAL_SOURCES_INTENDED
  StablehloOps.cpp
  StablehloBytecode.cpp

  DEPENDS
  StablehloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloAssemblyFormat
  StablehloBase
  StablehloTypeInference
  MLIRArithDialect
  MLIRDataLayoutInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSparseTensorDialect
  MLIRSupport
  MLIRTensorDialect
)

target_include_directories(StablehloOps INTERFACE
  $<BUILD_INTERFACE:${STABLEHLO_SOURCE_DIR}>
  $<BUILD_INTERFACE:${STABLEHLO_BINARY_DIR}>
)

set(LLVM_TARGET_DEFINITIONS VhloOps.td)
mlir_tablegen(VhloOps.h.inc -gen-op-decls)
mlir_tablegen(VhloOps.cpp.inc -gen-op-defs)
mlir_tablegen(VhloOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(VhloOpInterfaces.cpp.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS VhloEnums.td)
mlir_tablegen(VhloEnums.h.inc -gen-enum-decls)
mlir_tablegen(VhloEnums.cpp.inc -gen-enum-defs)
set(LLVM_TARGET_DEFINITIONS VhloOps.td)
mlir_tablegen(VhloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(VhloAttrs.cpp.inc -gen-attrdef-defs)
set(LLVM_TARGET_DEFINITIONS VhloAttrs.td)
mlir_tablegen(VhloAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(VhloAttrInterfaces.cpp.inc -gen-attr-interface-defs)
set(LLVM_TARGET_DEFINITIONS VhloTypes.td)
mlir_tablegen(VhloTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(VhloTypeInterfaces.cpp.inc -gen-type-interface-defs)
set(LLVM_TARGET_DEFINITIONS VhloOps.td)
mlir_tablegen(VhloTypeDefs.h.inc -gen-typedef-decls --typedefs-dialect=vhlo)
mlir_tablegen(VhloTypeDefs.cpp.inc -gen-typedef-defs --typedefs-dialect=vhlo)
add_public_tablegen_target(VhloOpsIncGen)
add_dependencies(mlir-headers VhloOpsIncGen)

add_mlir_dialect_library(VhloOps
  PARTIAL_SOURCES_INTENDED
  VhloBytecode.cpp
  VhloOps.cpp

  DEPENDS
  VhloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloAssemblyFormat
  VhloTypes
  VhloVersion
  MLIRIR
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSupport
)

add_mlir_dialect_library(VhloTypes
  PARTIAL_SOURCES_INTENDED
  VhloTypes.cpp

  DEPENDS
  VhloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloAssemblyFormat
  VhloVersion
  MLIRIR
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_dialect_library(VhloVersion
  PARTIAL_SOURCES_INTENDED
  Version.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
