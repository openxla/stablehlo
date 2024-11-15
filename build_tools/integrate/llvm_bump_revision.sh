#!/bin/bash
# Copyright 2024 The StableHLO Authors.
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

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
GH_ACTIONS="$SCRIPT_DIR/../github_actions"
REPO_ROOT="$SCRIPT_DIR/../.."

# Update build files
bump_to_xla_llvm_version() {
  LLVM_URL="https://raw.githubusercontent.com/openxla/xla/refs/heads/main/third_party/llvm/workspace.bzl"
  LLVM_REV=$(curl -s $LLVM_URL | grep 'LLVM_COMMIT =' | cut -d '"' -f 2)
  echo "Using LLVM commit: $LLVM_REV"
  echo "$LLVM_REV" > ./build_tools/llvm_version.txt
  "$GH_ACTIONS/lint_llvm_commit.sh" -f .
}

apply_xla_patch() {
  PATCH_URL="https://raw.githubusercontent.com/openxla/xla/refs/heads/main/third_party/stablehlo/temporary.patch"
  PATCH=$(curl -s "$PATCH_URL")
  if (( $(echo "$PATCH" | wc -l) < 2 )); then
    echo "Patch file openxla/xla/third_party/stablehlo/temporary.patch is empty"
    echo "Skipping patch apply"
    return 0
  fi

  TMP_DIR=$(mktemp -d)
  TMP_PATCH="$TMP_DIR/temporary.patch"
  echo "Cloning patch into $TMP_PATCH"
  echo "$PATCH" > "$TMP_PATCH"
  cd "$REPO_ROOT" || exit 1
  patch -p1 < "$TMP_PATCH"
}

bump_to_xla_llvm_version
apply_xla_patch

# Print the commit message
echo "Commit changes with message:"
echo "git add ."
echo "git commit -m \"Integrate LLVM at llvm/llvm-project@${LLVM_REV:0:12}\""
