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

add_mlir_library(StablehloReferenceApi
  PARTIAL_SOURCES_INTENDED
  Api.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
  InterpreterOps
  StablehloPasses
  StablehloReferenceConfiguration
  StablehloReferenceErrors
  StablehloReferenceNumPy
  StablehloReferenceOps
  StablehloReferenceProcess
  StablehloReferenceScope
  StablehloReferenceTensor
  StablehloReferenceValue
  StablehloRegister
)

add_mlir_library(StablehloReferenceAxes
  PARTIAL_SOURCES_INTENDED
  Axes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

add_mlir_library(StablehloReferenceConfiguration
  PARTIAL_SOURCES_INTENDED
  Configuration.cpp

  LINK_LIBS PUBLIC
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
  StablehloReferenceErrors
  StablehloReferenceProcess
  StablehloReferenceScope
)

add_mlir_library(StablehloReferenceElement
  PARTIAL_SOURCES_INTENDED
  Element.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
  StablehloReferenceTypes
)

add_mlir_library(StablehloReferenceErrors
  PARTIAL_SOURCES_INTENDED
  Errors.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

add_mlir_library(StablehloReferenceIndex
  PARTIAL_SOURCES_INTENDED
  Index.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

set(LLVM_TARGET_DEFINITIONS InterpreterOps.td)
mlir_tablegen(InterpreterOps.h.inc -gen-op-decls)
mlir_tablegen(InterpreterOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(InterpreterOpsIncGen)
add_dependencies(mlir-headers InterpreterOpsIncGen)

add_mlir_dialect_library(InterpreterOps
  PARTIAL_SOURCES_INTENDED
  InterpreterOps.cpp

  DEPENDS
  InterpreterOpsIncGen

  LINK_LIBS PUBLIC
  StablehloBase
  StablehloReferenceValue
  StablehloReferenceNumPy
  StablehloReferenceOps
  StablehloReferenceProcessGrid
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

set(LLVM_TARGET_DEFINITIONS InterpreterPasses.td)
mlir_tablegen(InterpreterPasses.h.inc -gen-pass-decls -name InterpreterTransforms)
add_public_tablegen_target(InterpreterTransformsPassIncGen)

add_mlir_library(InterpreterPasses
  PARTIAL_SOURCES_INTENDED
  InterpreterInstrumentWithProbe.cpp

  DEPENDS
  InterpreterTransformsPassIncGen

  LINK_LIBS PUBLIC
  InterpreterOps
  MLIRIR
  MLIRPass
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
  MLIRTransforms
  MLIRTransformUtils
)

add_mlir_dialect_library(StablehloReferenceNumPy
  PARTIAL_SOURCES_INTENDED
  NumPy.cpp

  LINK_LIBS PUBLIC
  StablehloReferenceTensor
  MLIRIR
)

add_mlir_library(StablehloReferenceOps
  PARTIAL_SOURCES_INTENDED
  Ops.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  StablehloBase
  StablehloOps
  StablehloReferenceAxes
  StablehloReferenceElement
  StablehloReferenceScope
  StablehloReferenceIndex
  StablehloReferenceValue
  StablehloReferenceProcess
  StablehloReferenceProcessGrid
  StablehloReferenceTensor
  StablehloReferenceToken
  StablehloTypeInference
)

add_mlir_library(StablehloReferenceProcess
  PARTIAL_SOURCES_INTENDED
  Process.cpp

  LINK_LIBS PUBLIC
  StablehloReferenceProcessGrid
  StablehloReferenceTensor
)

add_mlir_library(StablehloReferenceProcessGrid
  PARTIAL_SOURCES_INTENDED
  ProcessGrid.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
  StablehloReferenceTensor
)

add_mlir_library(StablehloReferenceScope
  PARTIAL_SOURCES_INTENDED
  Scope.cpp

  LINK_LIBS PUBLIC
  StablehloReferenceValue
  StablehloReferenceTensor
  StablehloReferenceToken
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

add_mlir_library(StablehloReferenceTensor
  PARTIAL_SOURCES_INTENDED
  Tensor.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  StablehloReferenceAxes
  StablehloReferenceElement
  StablehloReferenceErrors
  StablehloReferenceIndex
  StablehloReferenceTypes
)

add_mlir_library(StablehloReferenceToken
  PARTIAL_SOURCES_INTENDED
  Token.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
  StablehloOps
)

add_mlir_library(StablehloReferenceTypes
  PARTIAL_SOURCES_INTENDED
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

add_mlir_library(StablehloReferenceValue
  PARTIAL_SOURCES_INTENDED
  Value.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
  StablehloReferenceTensor
  StablehloReferenceToken
)
