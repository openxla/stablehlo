# Copyright 2022 OpenXLA Authors. All Rights Reserved.
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
#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StablehloTOSATransforms)
add_public_tablegen_target(StablehloTOSATransformsPassIncGen)

add_mlir_pdll_library(StablehloTOSAPDLLPatternsIncGen
  StablehloLegalizeToTosa.pdll
  StablehloLegalizeToTosa.pdll.h.inc
)

add_mlir_library(StablehloTOSATransforms
  StablehloLegalizeToTosa.cpp
  StablehloPrepareForTosa.cpp
  StablehloQuantLegalizeToTosaRescale.cpp
  TosaRescaleLegalizeToStablehlo.cpp

  DEPENDS
  StablehloTOSATransformsPassIncGen
  StablehloTOSAPDLLPatternsIncGen
  StablehloOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
)
