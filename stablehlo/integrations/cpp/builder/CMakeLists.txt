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


#####
# TableGen
#####

set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

add_tablegen(mlir_builder_tblgen MLIR_BUILDER
  EXPORT MLIR_BUILDER
  PARTIAL_SOURCES_INTENDED
  MlirBuilderTblgen.cpp
)

target_link_libraries(mlir_builder_tblgen
  PRIVATE
  MLIRTblgenLib
  MLIRSupport)

mlir_check_all_link_libraries(mlir_builder_tblgen)

include(MlirBuilderTblgen.cmake)

####
## Attr / Type Builder Helpers
####

add_mlir_library(AttrTypeBuilderUtil
  PARTIAL_SOURCES_INTENDED
  AttrTypeBuilderUtil.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

# cc_test(
#     name = "attr_type_builder_util_test",
#     srcs = ["stablehlo/integrations/cpp/builder/AttrTypeBuilderUtilTest.cpp"],
#     deps = [
#         ":attr_type_builder_util",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//third-party/unittest:gmock",
#         "@llvm-project//third-party/unittest:gtest",
#     ],
# )

####
## MlirBuilder base class
####

add_mlir_library(MlirBuilder
  PARTIAL_SOURCES_INTENDED
  MlirBuilder.cpp

  LINK_LIBS PUBLIC
  AttrTypeBuilderUtil
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

# cc_test(
#     name = "mlir_builder_test",
#     srcs = ["stablehlo/integrations/cpp/builder/MlirBuilderTest.cpp"],
#     deps = [
#         ":attr_type_builder_util",
#         ":chlo_builder",
#         ":func_builder",
#         ":mlir_builder",
#         ":register",
#         ":stablehlo_builder",
#         ":stablehlo_ops",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TosaDialect",
#         "@llvm-project//third-party/unittest:gmock",
#         "@llvm-project//third-party/unittest:gtest",
#     ],
# )

#####
## Dialect-specific builders
####

# gentbl_cc_library(
#     name = "chlo_builder_inc",
#     tbl_outs = {
#         "stablehlo/integrations/cpp/builder/ChloBuilder.h.inc": ["-gen-builder-decls"],
#         "stablehlo/integrations/cpp/builder/ChloBuilder.cpp.inc": ["-gen-builder-defs"],
#         "stablehlo/integrations/cpp/builder/ChloBuilder.md": ["-gen-builder-docs"],
#     },
#     tblgen = ":mlir_builder_tblgen",
#     td_file = ":stablehlo/dialect/ChloOps.td",
#     deps = [
#         ":chlo_ops_td_files",
#         "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
#         "@llvm-project//mlir:OpBaseTdFiles",
#         "@llvm-project//mlir:SideEffectInterfacesTdFiles",
#     ],
# )

set(LLVM_TARGET_DEFINITIONS "../../../dialect/ChloOps.td")
mlir_builder_tblgen(ChloBuilder.h.inc -gen-builder-decls)
mlir_builder_tblgen(ChloBuilder.cpp.inc -gen-builder-defs)
add_public_tablegen_target(ChloBuilderIncGen)
add_dependencies(mlir-headers ChloBuilderIncGen)

add_mlir_library(ChloBuilder
  PARTIAL_SOURCES_INTENDED
  ChloBuilder.cpp

  DEPENDS
  ChloBuilderIncGen

  LINK_LIBS PUBLIC
  ChloOps
  MlirBuilder
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

set(TODO_MLIR_PATH "/usr/local/google/home/gleasonk/Coding/llvm-project/mlir")
set(LLVM_TARGET_DEFINITIONS "${TODO_MLIR_PATH}/include/mlir/Dialect/Func/IR/FuncOps.td")
mlir_builder_tblgen(FuncBuilder.h.inc -gen-builder-decls)
mlir_builder_tblgen(FuncBuilder.cpp.inc -gen-builder-defs)
add_public_tablegen_target(FuncBuilderIncGen)
add_dependencies(mlir-headers FuncBuilderIncGen)

add_mlir_library(FuncBuilder
  PARTIAL_SOURCES_INTENDED
  FuncBuilder.cpp

  DEPENDS
  FuncBuilderIncGen

  LINK_LIBS PUBLIC
  MlirBuilder
  MLIRFuncDialect
  MLIRIR
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

set(LLVM_TARGET_DEFINITIONS "../../../dialect/StablehloOps.td")
mlir_builder_tblgen(StablehloBuilder.h.inc -gen-builder-decls)
mlir_builder_tblgen(StablehloBuilder.cpp.inc -gen-builder-defs)
add_public_tablegen_target(StablehloBuilderIncGen)
add_dependencies(mlir-headers StablehloBuilderIncGen)

add_mlir_library(StablehloBuilder
  PARTIAL_SOURCES_INTENDED
  StablehloBuilder.cpp

  DEPENDS
  StablehloBuilderIncGen

  LINK_LIBS PUBLIC
  AttrTypeBuilderUtil
  MlirBuilder
  StablehloOps
  StablehloTypeInference
  MLIRFuncDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSupport  # LLVMSupport is automatically added in AddLLVM.cmake.
)

if (TARGET llvm_gtest)
    set_target_properties(check-stablehlo-ci PROPERTIES FOLDER "Tests")
    add_unittest(check-stablehlo-ci "unittests"
      MlirBuilderTest.cpp
      StablehloBuilderTest.cpp
      AttrTypeBuilderUtilTest.cpp
    )
  else()
    message(WARNING "gtest not found, unittests will not be available")
  endif()

