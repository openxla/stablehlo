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

add_mlir_library(StablehloReferenceAxes
  PARTIAL_SOURCES_INTENDED
  Axes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_library(StablehloReferenceElement
  PARTIAL_SOURCES_INTENDED
  Element.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  StablehloReferenceTypes
)

add_mlir_library(StablehloReferenceIndex
  PARTIAL_SOURCES_INTENDED
  Index.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
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
  StablehloReferenceInterpreterValue
  StablehloReferenceNumPy
  StablehloReferenceOps
  StablehloReferenceProcessGrid
  MLIRIR
  MLIRSupport
)

add_mlir_library(StablehloReferenceInterpreterValue
  PARTIAL_SOURCES_INTENDED
  InterpreterValue.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  StablehloReferenceTensor
  StablehloReferenceToken
)

add_mlir_dialect_library(StablehloReferenceNumPy
  PARTIAL_SOURCES_INTENDED
  NumPy.cpp

  LINK_LIBS PUBLIC
  StablehloReferenceIndex
  StablehloReferenceTensor
  MLIRIR
  MLIRSupport
)

add_mlir_library(StablehloReferenceOps
  PARTIAL_SOURCES_INTENDED
  Ops.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  StablehloOps
  StablehloReferenceAxes
  StablehloReferenceElement
  StablehloReferenceScope
  StablehloReferenceIndex
  StablehloReferenceInterpreterValue
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
  MLIRSupport
  StablehloReferenceTensor
)

add_mlir_library(StablehloReferenceScope
  PARTIAL_SOURCES_INTENDED
  Scope.cpp

  LINK_LIBS PUBLIC
  StablehloReferenceInterpreterValue
  StablehloReferenceTensor
  StablehloReferenceToken
  MLIRSupport
)

add_mlir_library(StablehloReferenceTensor
  PARTIAL_SOURCES_INTENDED
  Tensor.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  StablehloReferenceAxes
  StablehloReferenceElement
  StablehloReferenceIndex
  StablehloReferenceTypes
)

add_mlir_library(StablehloReferenceToken
  PARTIAL_SOURCES_INTENDED
  Token.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  StablehloOps
)

add_mlir_library(StablehloReferenceTypes
  PARTIAL_SOURCES_INTENDED
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
