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
# caching in GitHub Actions to improve build speeds.

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <path/to/llvm> <build_dir>"
  exit 1
fi

# LLVM source
LLVM_SRC_DIR="$1"
LLVM_BUILD_DIR="$2"

# Configure LLVM
cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$LLVM_BUILD_DIR" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache

# Build LLVM/MLIR
cmake --build "$LLVM_BUILD_DIR" --target all
