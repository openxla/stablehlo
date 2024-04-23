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

set -o errexit
set -o nounset
set -o pipefail

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <llvm_build_dir> <stablehlo_build_dir>"
  exit 1
fi

LLVM_BUILD_DIR="$1"
STABLEHLO_BUILD_DIR="$2"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

# Turn on building Python bindings
STABLEHLO_ENABLE_BINDINGS_PYTHON="${STABLEHLO_ENABLE_BINDINGS_PYTHON:-OFF}"
# Turn on running SavedModel Python tests requiring TF dependency
STABLEHLO_ENABLE_PYTHON_TF_TESTS="${STABLEHLO_ENABLE_PYTHON_TF_TESTS:-OFF}"
# Turn on building Sanitizers
# Note: This is not congruent with building python bindings
STABLEHLO_ENABLE_SANITIZER="${STABLEHLO_ENABLE_SANITIZER:-OFF}"

# Configure StableHLO
# CMAKE_PLATFORM_NO_VERSIONED_SONAME Disables generation of "version soname"
#                         (i.e. libFoo.so.<version>), which causes pure
#                         duplication of various shlibs for Python wheels.
cmake -GNinja \
  -B"$STABLEHLO_BUILD_DIR" \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DSTABLEHLO_ENABLE_STRICT_BUILD=ON \
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME:BOOL=ON \
  -DSTABLEHLO_ENABLE_SANITIZER="$STABLEHLO_ENABLE_SANITIZER" \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON="$STABLEHLO_ENABLE_BINDINGS_PYTHON" \
  -DSTABLEHLO_ENABLE_PYTHON_TF_TESTS="$STABLEHLO_ENABLE_PYTHON_TF_TESTS"

# Build and Test StableHLO
cd "$STABLEHLO_BUILD_DIR" || exit
ninja check-stablehlo-ci
