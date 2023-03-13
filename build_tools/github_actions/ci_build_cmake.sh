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

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <llvm_build_dir> <stablehlo_build_dir>"
  exit 1
fi

LLVM_BUILD_DIR="$1"
STABLEHLO_BUILD_DIR="$2"

# Configure StableHLO
cmake -GNinja \
  -B"$STABLEHLO_BUILD_DIR" \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DSTABLEHLO_ENABLE_STRICT_BUILD=On

# Build and Test StableHLO
cd "$STABLEHLO_BUILD_DIR"
ninja check-stablehlo
