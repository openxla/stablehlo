# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#
cmake_minimum_required(VERSION 3.15.0)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# Support for return(PROPAGATE ...) in functions.
if (POLICY CMP0140)
  cmake_policy(SET CMP0140 NEW)
endif()

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------
option(STABLEHLO_BUILD_EMBEDDED "Build StableHLO as part of another project" OFF)
option(STABLEHLO_ENABLE_BINDINGS_PYTHON "Enables StableHLO Python bindings" OFF)
option(STABLEHLO_ENABLE_PYTHON_TF_TESTS "Enables StableHLO to SavedModel tests requiring TF" OFF)
option(STABLEHLO_ENABLE_STRICT_BUILD "Build StableHLO with strict warnings and warnings as errors" OFF)
option(STABLEHLO_ENABLE_SANITIZER "Enable a sanitizer [OFF, address]" OFF)
option(STABLEHLO_ENABLE_SPLIT_DWARF "Enable split DWARF if the platform supports it" OFF)
option(STABLEHLO_ENABLE_LLD "Use LLD as the linker if available" OFF)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

# There are 3 build modes, one will be set to ON
#  - Standalone: Build MLIR as a part of StableHLO, requires registering LLVM globals
#  - External: StableHLO built as an external LLVM project (XLA/MHLO uses this)
#  - Embedded: StableHLO built as a part of another MLIR project (torch-mlir uses this)
#
# If building as part of another project, let it handle the MLIR dependency.
# The dependent project might use a bundled version of MLIR instead of installing.
set(STABLEHLO_EXTERNAL_PROJECT_BUILD OFF)
set(STABLEHLO_STANDALONE_BUILD OFF)
# STABLEHLO_BUILD_EMBEDDED declared in above options

if (NOT STABLEHLO_BUILD_EMBEDDED)
  if ((CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) OR MLIR_BINARY_DIR)
    set(STABLEHLO_STANDALONE_BUILD ON)
  else()
    set(STABLEHLO_EXTERNAL_PROJECT_BUILD ON)
  endif()
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
if (STABLEHLO_ENABLE_STRICT_BUILD)
  set(LLVM_ENABLE_WARNINGS ON)
  set(LLVM_ENABLE_WERROR ON)
  set(LLVM_ENABLE_PEDANTIC ON)
endif()

if(STABLEHLO_EXTERNAL_PROJECT_BUILD)
  message(STATUS "Building StableHLO as an external LLVM project")
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_GENERATED_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
endif()

if(STABLEHLO_STANDALONE_BUILD)
  message("Building StableHLO as a standalone project.")
  project(stablehlo LANGUAGES CXX C)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_CXX_STANDARD 17)

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(HandleLLVMOptions)
endif()

if(STABLEHLO_BUILD_EMBEDDED)
  message(STATUS "Building StableHLO embedded in another project")
  include(HandleLLVMOptions)
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Add the CMake modules specific to StableHLO
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(LLVM_ENABLE_ZLIB)
  find_package(ZLIB)
endif()

#-------------------------------------------------------------------------------
# Compiler options
#-------------------------------------------------------------------------------

if(MSVC)
  # Required to use M_PI. Must be set before any `#include <cmath>`.
  add_compile_definitions("_USE_MATH_DEFINES")

  # Tablegen outputs from LLVM/MLIR can easily exceed the default 64k section
  # count limit, so increase to 2^32.
  add_compile_options("/bigobj")
endif()

#-------------------------------------------------------------------------------
# Performance configuration
#-------------------------------------------------------------------------------

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
if (STABLEHLO_ENABLE_LLD)
  message(STATUS "Enabling LLD as the linker")
  add_link_options("-fuse-ld=lld")
endif()

if(STABLEHLO_ENABLE_SPLIT_DWARF)
    check_cxx_compiler_flag(-gsplit-dwarf STABLEHLO_SUPPORTS_SPLIT_DWARF)
    if (STABLEHLO_SUPPORTS_SPLIT_DWARF)
      message(STATUS "Enabling split-dwarf build")
      add_compile_options(-gsplit-dwarf -ggnu-pubnames)
    endif()
    check_linker_flag(CXX "-Wl,--gdb-index" STABLEHLO_SUPPORTS_GDB_INDEX)
    # If we set LLD it doesn't seem to affect the check_linker_flag above.
    # Account for it with the generator expression OR
    if (STABLEHLO_SUPPORTS_GDB_INDEX OR STABLEHLO_ENABLE_LLD)
      message(STATUS "Enabling GDB index in binary")
      add_link_options("-Wl,--gdb-index")
    endif()
endif()

#TODO: Where should these be?
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


#-------------------------------------------------------------------------------
# Sanitizer configuration
#-------------------------------------------------------------------------------

include(SetupSanitizers)
setup_sanitizers()

#-------------------------------------------------------------------------------
# Python configuration
#-------------------------------------------------------------------------------

if(STABLEHLO_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

set(STABLEHLO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(STABLEHLO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(STABLEHLO_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

# Target that provides comprehensive testing for the StableHLO repository.
# It includes both quick and slow tests (see check-stablehlo-quick and
# check-stablehlo-slow below).
add_custom_target(check-stablehlo-ci)

# Target that aggregates slow tests. Long-running test targets should go
# into dependencies of this target, and we shouldn't expect humans to run them
# regularly during development.
add_custom_target(check-stablehlo-slow)
add_dependencies(check-stablehlo-ci check-stablehlo-slow)

# Target that aggregates fast tests. We shouldn't add long-running test targets
# as a dependency of this target, since we'd want humans to routinely use this
# target during development.
add_custom_target(check-stablehlo-quick)
add_dependencies(check-stablehlo-ci check-stablehlo-quick)

add_subdirectory(stablehlo)
add_subdirectory(examples)
