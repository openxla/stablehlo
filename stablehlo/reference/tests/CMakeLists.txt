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

#include(FetchContent)
#
#FetchContent_Declare(
#  googletest
#  GIT_REPOSITORY https://github.com/google/googletest.git
#  GIT_TAG        release-1.11.0
#)
#FetchContent_MakeAvailable(googletest)
#add_library(GTest::GTest INTERFACE IMPORTED)
#target_link_libraries(GTest::GTest INTERFACE gtest_main)
#
#add_mlir_library(TestUtils
#  PARTIAL_SOURCES_INTENDED
#  TestUtils.cpp
#
#  LINK_LIBS PUBLIC
#  GTest::GTest
#  MLIRAsmParser
#  MLIRFuncDialect
#  MLIRIR
#  MLIRParser
#  MLIRSupport
#  MLIRDataLayoutInterfaces
#  StablehloOps
#  StablehloReferenceInterpreter
#  StablehloReferenceTensor
#)
#
#set(TESTS
#  AddOpTest
#  # more targets ...
#)
#
#set(TEST_LIBS
#  GTest::GTest
#  TestUtils
#)
#
#include(GoogleTest)
#
#foreach(TEST ${TESTS})
#    add_executable(${TEST} ${TEST}.cpp)
#    target_link_libraries(${TEST} PRIVATE ${TEST_LIBS})
#    gtest_discover_tests(${TEST})
#endforeach()
#
#add_custom_target(check-stablehlo-interpreter ${TESTS})
#add_dependencies(check-stablehlo check-stablehlo-interpreter)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(STABLEHLO_INTERPRETER_TEST_DEPENDS
        FileCheck count not
        stablehlo-interpreter-runner
)
add_lit_testsuite(check-stablehlo-interpreter-lit "Running the StableHLO interpreter unit tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${STABLEHLO_INTERPRETER_TEST_DEPENDS}
        )

set_target_properties(check-stablehlo-interpreter-lit PROPERTIES FOLDER "Tests")
add_lit_testsuites(STABLEHLO_INTERPRETER_SUITE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${STABLEHLO_INTERPRETER_TEST_DEPENDS})
add_dependencies(check-stablehlo check-stablehlo-interpreter-lit)

add_llvm_executable(stablehlo-interpreter-runner StablehloInterpreterRunner.cpp)
llvm_update_compile_flags(stablehlo-interpreter-runner)
target_link_libraries(stablehlo-interpreter-runner PRIVATE
  MLIRAsmParser
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRDataLayoutInterfaces
  StablehloOps
  StablehloReferenceInterpreter
  StablehloReferenceTensor
)

mlir_check_all_link_libraries(stablehlo-interpreter-runner)
