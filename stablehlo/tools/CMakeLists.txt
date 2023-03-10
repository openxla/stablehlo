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

set(LLVM_OPTIONAL_SOURCES
  StablehloOptMain.cpp
  StablehloTranslateMain.cpp
  Interpreter.cpp
)

# stablehlo-opt
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        StablehloRegister
        StablehloTestUtils
        StablehloPasses
        )
add_llvm_executable(stablehlo-opt StablehloOptMain.cpp)
llvm_update_compile_flags(stablehlo-opt)
target_link_libraries(stablehlo-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(stablehlo-opt)

# stablehlo-translate
add_llvm_executable(stablehlo-translate StablehloTranslateMain.cpp)
llvm_update_compile_flags(stablehlo-translate)
target_link_libraries(stablehlo-translate PRIVATE
  ${dialect_libs}
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  CheckOps
  StablehloOps
  StablehloReferenceOps
  StablehloReferenceScope
  StablehloReferenceTensor
  StablehloRegister
  StablehloSerialization
  VhloOps
)

mlir_check_all_link_libraries(stablehlo-translate)
