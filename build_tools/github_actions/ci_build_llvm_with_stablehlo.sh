#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# This file is similar to build_mlir.sh, but passes different flags for
# caching in GitHub Actions.

# This file gets called on build directory where resources are placed
# during `ci_configure`, and builds stablehlo in the directory specified
# by the second argument.

if [[ $# -ne 3 ]] ; then
  echo "Usage: $0 <llvm_project_dir> <llvm_build_dir> <stablehlo_project_dir>"
  exit 1
fi

LLVM_PROJ_DIR="$1"
LLVM_BUILD_DIR="$2"
STABLEHLO_PROJ_DIR="$3"

# Configure StableHLO
cmake -GNinja \
  -B"$LLVM_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_WARNINGS=ON \
  -DLLVM_ENABLE_WERROR=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_EXTERNAL_PROJECTS="stablehlo" \
  -DLLVM_EXTERNAL_STABLEHLO_SOURCE_DIR="$STABLEHLO_PROJ_DIR" \
  $LLVM_PROJ_DIR/llvm

# Build and Check StableHLO
cd "$LLVM_BUILD_DIR"
cmake --build . --target stablehlo-opt --target stablehlo-interpreter
