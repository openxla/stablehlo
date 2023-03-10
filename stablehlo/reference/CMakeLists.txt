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
  Sizes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_library(StablehloReferenceSizes
  PARTIAL_SOURCES_INTENDED
  Sizes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_library(StablehloReferenceTypes
  PARTIAL_SOURCES_INTENDED
  Types.cpp

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
  StablehloReferenceSizes
)

add_mlir_library(StablehloReferenceScope
  PARTIAL_SOURCES_INTENDED
  Scope.cpp

  LINK_LIBS PUBLIC
  StablehloReferenceTensor
  MLIRSupport
)

add_mlir_library(StablehloReferenceTensor
  PARTIAL_SOURCES_INTENDED
  Tensor.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  StablehloReferenceElement
  StablehloReferenceIndex
  StablehloReferenceSizes
  StablehloReferenceTypes
)

add_mlir_library(StablehloReferenceOps
  PARTIAL_SOURCES_INTENDED
  Ops.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  StablehloOps
  StablehloReferenceAxes
  StablehloReferenceElement
  StablehloReferenceIndex
  StablehloReferenceScope
  StablehloReferenceSizes
  StablehloReferenceTensor
)
