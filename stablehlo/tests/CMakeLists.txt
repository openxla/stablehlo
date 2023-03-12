# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
add_lit_testsuite(check-stablehlo-tests "Running the tests/ suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  FileCheck
  stablehlo-opt
  stablehlo-translate
)
add_dependencies(check-stablehlo check-stablehlo-tests)

set(LLVM_TARGET_DEFINITIONS TestUtils.td)
mlir_tablegen(TestUtils.h.inc -gen-pass-decls -name HloTest)
add_public_tablegen_target(StablehloTestUtilsIncGen)

add_mlir_library(StablehloTestUtils
  PARTIAL_SOURCES_INTENDED
  TestUtils.cpp

  DEPENDS
  StablehloTestUtilsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRPass
  MLIRShapeDialect
  MLIRSupport
  MLIRTransformUtils
  StablehloAssemblyFormat
)

set(LLVM_TARGET_DEFINITIONS CheckOps.td)
mlir_tablegen(CheckOps.h.inc -gen-op-decls)
mlir_tablegen(CheckOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CheckOpsIncGen)
add_dependencies(mlir-headers CheckOpsIncGen)

add_mlir_dialect_library(CheckOps
  PARTIAL_SOURCES_INTENDED
  CheckOps.cpp

  DEPENDS
  CheckOpsIncGen

  LINK_LIBS PUBLIC
  StablehloBase
  StablehloReferenceTensor
  MLIRIR
  MLIRSupport
)
