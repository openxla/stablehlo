#!/bin/bash
# Copyright 2022 The StableHLO Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [[ $# -ne 3 ]] ; then
  echo "Usage: $0 <llvm_project_dir> <stablehlo_api_build_dir> <stablehlo_root_dir>"
  exit 1
fi

LLVM_PROJECT_DIR="$1"
STABLEHLO_PYTHON_BUILD_DIR="$2"
STABLEHLO_ROOT_DIR="$3"

# Configure StableHLO Python Bindings
cmake -GNinja \
  -B"$STABLEHLO_PYTHON_BUILD_DIR" \
  $LLVM_PROJECT_DIR/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=stablehlo \
  -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR="$STABLEHLO_ROOT_DIR" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DPython3_EXECUTABLE=$(which python3) \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DSTABLEHLO_ENABLE_STRICT_BUILD=On

# Build and Check StableHLO Python Bindings
cd "$STABLEHLO_PYTHON_BUILD_DIR"
ninja check-stablehlo-python
