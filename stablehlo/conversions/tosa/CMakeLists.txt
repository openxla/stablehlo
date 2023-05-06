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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check-stablehlo-tosa)

add_subdirectory(transforms)
add_subdirectory(tests)

add_mlir_pdll_library(StablehloTOSAPDLLPatternsIncGen
  transforms/legalize_stablehlo.pdll
  transforms/legalize_stablehlo.pdll.h.inc
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib

        MLIRTosaDialect
	StablehloRegister

        StablehloTOSATransforms
        )
add_llvm_executable(stablehlo-tosa-opt stablehlo_tosa_opt.cc
)
llvm_update_compile_flags(stablehlo-tosa-opt)
add_dependencies(stablehlo-tosa-opt StablehloTOSATransformsPassIncGen)
target_link_libraries(stablehlo-tosa-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(stablehlo-tosa-opt)
