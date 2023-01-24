#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2023 The StableHLO Authors.
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

set -e

if [[ $# -ne 1 ]] ; then
  echo "Usage: $0 <bazel_workspace_dir>"
  exit 1
fi

WORKSPACE_DIR="$1"

#LLVM version
set -x
echo "=== Retrieving LLVM Commit & Calculating SHA256 ==="
export LLVM_COMMIT="$(cat $WORKSPACE_DIR/build_tools/llvm_version.txt)"
echo "LLVM_COMMIT: $LLVM_COMMIT"
export LLVM_SHA256="$(curl -sL https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz | shasum -a 256 | sed 's/ //g; s/-//g')"
echo "LLVM_SHA256: $LLVM_SHA256"

cd $WORKSPACE_DIR
sed -i '/^LLVM_COMMIT/s/"[^"]*"/"'$LLVM_COMMIT'"/g' WORKSPACE
sed -i '/^LLVM_SHA256/s/"[^"]*"/"'$LLVM_SHA256'"/g' WORKSPACE

# Build StableHLO
echo "=== Building StableHLO ==="
bazel build //:all
set +x
