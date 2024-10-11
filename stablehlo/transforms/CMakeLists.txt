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

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(PassesIncGen)

set(LLVM_TARGET_DEFINITIONS ChloDecompositionPatterns.td)
mlir_tablegen(ChloDecompositionPatterns.h.inc --gen-rewriters)
add_public_tablegen_target(ChloDecompositionPatternsIncGen)

set(LLVM_TARGET_DEFINITIONS StablehloAggressiveSimplificationPatterns.td)
mlir_tablegen(StablehloAggressiveSimplificationPatterns.h.inc --gen-rewriters)
add_public_tablegen_target(StablehloAggressiveSimplificationPatternsIncGen)

set(LLVM_TARGET_DEFINITIONS StablehloCompatibilityExpanderPatterns.td)
mlir_tablegen(StablehloCompatibilityExpanderPatterns.h.inc --gen-rewriters)
add_public_tablegen_target(StablehloCompatibilityExpanderPatternsIncGen)

set(LLVM_TARGET_DEFINITIONS StablehloLegalizeDeprecatedOpsPatterns.td)
mlir_tablegen(StablehloLegalizeDeprecatedOpsPatterns.h.inc --gen-rewriters)
add_public_tablegen_target(StablehloLegalizeDeprecatedOpsPatternsIncGen)

set(LLVM_TARGET_DEFINITIONS VhloToVersionPatterns.td)
mlir_tablegen(VhloToVersionPatterns.h.inc --gen-rewriters)
add_public_tablegen_target(VhloToVersionPatterns)


add_mlir_dialect_library(StablehloPasses
  PARTIAL_SOURCES_INTENDED
  ChloLegalizeToStablehlo.cpp
  PassPipelines.cpp
  ShapeLegalizeToStablehlo.cpp
  StablehloAggressiveFolder.cpp
  StablehloAggressiveSimplification.cpp
  StablehloCanonicalizeDynamism.cpp
  StablehloConvertToSignless.cpp
  StablehloCompatibilityExpander.cpp
  StablehloLegalizeCompositeToCall.cpp
  StablehloLegalizeDeprecatedOps.cpp
  StablehloLegalizeQuantToMath.cpp
  StablehloLegalizeQuantizedOpToQDQ.cpp
  StablehloLegalizeQDQToQuantizedOp.cpp
  StablehloLegalizeToVhlo.cpp
  StablehloRefineArguments.cpp
  StablehloRefineShapes.cpp
  VhloLegalizeToStablehlo.cpp
  VhloToVersion.cpp
  PassUtils.cpp

  DEPENDS
  ChloDecompositionPatternsIncGen
  PassesIncGen
  StablehloAggressiveSimplificationPatternsIncGen
  StablehloCompatibilityExpanderPatternsIncGen
  StablehloLegalizeDeprecatedOpsPatternsIncGen
  VhloToVersionPatterns

  LINK_LIBS PUBLIC
  ChloOps
  MLIRArithDialect
  MLIRAsmParser
  MLIRComplexDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRFunctionInterfaces
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRQuantDialect
  MLIRReconcileUnrealizedCasts
  MLIRSupport
  MLIRTensorDialect
  MLIRTransformUtils
  StablehloBase
  StablehloBroadcastUtils
  StablehloLinalgTransforms
  StablehloOps
  StablehloTypeInference
  VhloOps
)
