# Copyright 2025 The StableHLO Authors.
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

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name=Optimization)
add_public_tablegen_target(OptimizationPassesIncGen)

set(LLVM_TARGET_DEFINITIONS StablehloAggressiveSimplificationPatterns.td)
mlir_tablegen(StablehloAggressiveSimplificationPatterns.h.inc --gen-rewriters)
add_public_tablegen_target(StablehloAggressiveSimplificationPatternsIncGen)

add_mlir_dialect_library(StablehloOptimizationPasses
  PARTIAL_SOURCES_INTENDED
  StablehloAggressiveFolder.cpp
  StablehloAggressiveSimplification.cpp
  StablehloTargetIndependentOptimization.cpp

  DEPENDS
  OptimizationPassesIncGen
  StablehloAggressiveSimplificationPatternsIncGen
  StablehloPassUtils

  LINK_LIBS PUBLIC
  ChloOps
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRRewrite
  MLIRSupport
  MLIRTransformUtils
  StablehloBase
  StablehloOps
  StablehloTypeInference
  StablehloPassUtils
)
