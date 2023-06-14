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

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------
option(STABLEHLO_BUILD_EMBEDDED "Build StableHLO as part of another project" OFF)
option(STABLEHLO_ENABLE_BINDINGS_PYTHON "Enables StableHLO Python bindings" OFF)
option(STABLEHLO_ENABLE_STRICT_BUILD "Build StableHLO with strict warnings and warnings as errors" OFF)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------
set(STABLEHLO_EXTERNAL_PROJECT_BUILD OFF)

if(NOT (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) AND NOT MLIR_BINARY_DIR)
  # Building as part of LLVM via the external project mechanism.
  set(STABLEHLO_EXTERNAL_PROJECT_BUILD ON)
else()
  # Building standalone.
  project(stablehlo LANGUAGES CXX C)
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Build with ccache if the package is present
set(LLVM_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(LLVM_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
      set(LLVM_CCACHE_MAXSIZE "" CACHE STRING "Size of ccache")
      set(LLVM_CCACHE_DIR "" CACHE STRING "Directory to keep ccached data")
      set(LLVM_CCACHE_PARAMS "CCACHE_CPP2=yes CCACHE_HASHDIR=yes"
          CACHE STRING "Parameters to pass through to ccache")

      set(CCACHE_PROGRAM "${LLVM_CCACHE_PARAMS} ${CCACHE_PROGRAM}")
      if (LLVM_CCACHE_MAXSIZE)
        set(CCACHE_PROGRAM "CCACHE_MAXSIZE=${LLVM_CCACHE_MAXSIZE} ${CCACHE_PROGRAM}")
      endif()
      if (LLVM_CCACHE_DIR)
        set(CCACHE_PROGRAM "CCACHE_DIR=${LLVM_CCACHE_DIR} ${CCACHE_PROGRAM}")
      endif()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(FATAL_ERROR "Unable to find the program ccache. Set LLVM_CCACHE_BUILD to OFF")
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

# Find MLIR to install if we are building standalone. If building as part of
# another project, let it handle the MLIR dependency. The dependent project
# might use a bundled version of MLIR instead of installing, for instance.
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
elseif(NOT STABLEHLO_BUILD_EMBEDDED)
  message(STATUS "Building StableHLO with an installed MLIR")
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
else()
  message(STATUS "Building StableHLO embedded in another project")
endif()

if(LLVM_ENABLE_ZLIB)
  find_package(ZLIB)
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Python configuration
#-------------------------------------------------------------------------------

if(STABLEHLO_ENABLE_BINDINGS_PYTHON)
  if(NOT STABLEHLO_EXTERNAL_PROJECT_BUILD)
    message(WARNING "StableHLO Python bindings are not supported in standalone mode")
  endif()

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
