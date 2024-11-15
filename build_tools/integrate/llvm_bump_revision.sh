#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
GH_ACTIONS="$SCRIPT_DIR/../github_actions/"
REPO_ROOT="$SCRIPT_DIR/../.."

# Update build files
bump_to_xla_llvm_version() {
  LLVM_URL="https://raw.githubusercontent.com/openxla/xla/refs/heads/main/third_party/llvm/workspace.bzl"
  LLVM_REV=$(curl -s LLVM_URL | grep 'LLVM_COMMIT =' | cut -d '"' -f 2)
  echo "Bumping to LLVM commit: $LLVM_REV"
  echo "$LLVM_REV" > ./build_tools/llvm_version.txt

  # Lint revision
  SCRIPT_DIR="$(dirname "$(realpath "$0")")"
  lint_llvm_commit.sh -f .
}

apply_xla_patch() {
  PATCH_URL="https://raw.githubusercontent.com/openxla/xla/refs/heads/main/third_party/stablehlo/temporary.patch"
  PATCH=$(curl -s $PATCH_URL)
  if (( $(echo "$PATCH" | wc -l) < 2 )); then
    echo "Patch file is empty, skipping patch."
    return 0
  fi

  TMP_DIR=$(mktemp -d)
  TMP_PATCH="$TMP_DIR/temporary.patch"
  echo "Cloning patch into $TMP_PATCH"
  echo "$PATCH" > $TMP_PATCH
  cd $REPO_ROOT
  patch -p1 < $TMP_PATCH
}

bump_to_xla_llvm_version
apply_xla_patch

# Print the commit message
echo "Commit changes with message:"
echo "git add ."
echo "git commit -m \"Integrate LLVM at llvm/llvm-project@${LLVM_REV:0:12}\""
