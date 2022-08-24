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
include(AddMLIRPython)

################################################################################
# Generate Python sources
################################################################################

declare_mlir_python_sources(ChloPythonSources)
declare_mlir_python_sources(ChloPythonSources.Dialects
  ADD_TO_PARENT ChloPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ChloPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/ChloOps.td
  SOURCES dialects/chlo.py
  DIALECT_NAME chlo)

declare_mlir_python_sources(StablehloPythonSources)
declare_mlir_python_sources(StablehloPythonSources.Dialects
  ADD_TO_PARENT StablehloPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT StablehloPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/StablehloOps.td
  SOURCES dialects/stablehlo.py
  DIALECT_NAME stablehlo)

################################################################################
# Generate Python extensions
################################################################################

declare_mlir_python_sources(ChloPythonExtensions)
declare_mlir_python_extension(ChloPythonExtensions.Main
  MODULE_NAME _chlo
  ADD_TO_PARENT ChloPythonExtensions
  SOURCES
    ChloModule.cpp
  EMBED_CAPI_LINK_LIBS
    ChloCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_sources(StablehloPythonExtensions)
declare_mlir_python_extension(StablehloPythonExtensions.Main
  MODULE_NAME _stablehlo
  ADD_TO_PARENT StablehloPythonExtensions
  SOURCES
    StablehloModule.cpp
  EMBED_CAPI_LINK_LIBS
    StablehloCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared libraries
################################################################################

add_mlir_python_common_capi_library(ChloPythonCAPI
  INSTALL_COMPONENT ChloPythonModules
  INSTALL_DESTINATION python_packages/chlo/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${STABLEHLO_BINARY_DIR}/python_packages/chlo/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    ChloPythonSources
    ChloPythonExtensions
)

add_mlir_python_modules(ChloPythonModules
  ROOT_PREFIX "${STABLEHLO_BINARY_DIR}/python_packages/chlo/mlir"
  INSTALL_PREFIX "python_packages/chlo/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    ChloPythonSources
    ChloPythonExtensions
  COMMON_CAPI_LINK_LIBS
    ChloPythonCAPI
  )

add_mlir_python_common_capi_library(StablehloPythonCAPI
  INSTALL_COMPONENT StablehloPythonModules
  INSTALL_DESTINATION python_packages/stablehlo/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${STABLEHLO_BINARY_DIR}/python_packages/stablehlo/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    StablehloPythonSources
    StablehloPythonExtensions
)

add_mlir_python_modules(StablehloPythonModules
  ROOT_PREFIX "${STABLEHLO_BINARY_DIR}/python_packages/stablehlo/mlir"
  INSTALL_PREFIX "python_packages/stablehlo/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    StablehloPythonSources
    StablehloPythonExtensions
  COMMON_CAPI_LINK_LIBS
    StablehloPythonCAPI
  )
