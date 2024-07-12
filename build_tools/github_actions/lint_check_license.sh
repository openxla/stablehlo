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

set -o errexit
set -o nounset
set -o pipefail

exit_with_usage() {
  echo "Usage: $0 [-b]"
  echo "    -b <branch>  Base branch name, defaults to main."
  exit 1
}

BASE_BRANCH=main
while getopts 'b:' flag; do
  case "${flag}" in
    b) BASE_BRANCH="$OPTARG" ;;
    *) exit_with_usage ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $# -ne 0 ]] ; then
  exit_with_usage
fi

echo "Gathering changed files..."
echo
mapfile -t CHANGED_FILES < <(git diff "$BASE_BRANCH" HEAD --name-only --diff-filter=d)
if (( ${#CHANGED_FILES[@]} == 0 )); then
  echo "Found no changed files."
  exit 0
fi

echo "Checking the following files for licenses:
$(printf "%s\n" "${CHANGED_FILES[@]}")"
echo

SKIPPED_SUFFIXES=(
  .clang-format
  .gitignore
  .ipynb
  .json
  .md
  .mlir
  .mlir.bc
  .png
  .patch
  .svg
  LICENSE
  MODULE.bazel.lock
  llvm_version.txt
)

UNLICENSED_FILES=()
for file in "${CHANGED_FILES[@]}"; do
  skip=0
  for suffix in "${SKIPPED_SUFFIXES[@]}"; do
    if [[ "$file" = *$suffix ]]; then
      skip=1
      break
    fi
  done
  if (( skip )); then
    echo "Skipping file: $file"
    continue;
  fi
  if ! head -20 "$file" | grep -Pzo '(?s)Copyright 20[1-9][0-9].*?Apache License, Version 2\.0' &>/dev/null; then
    UNLICENSED_FILES+=("$file")
  fi
done

if (( ${#UNLICENSED_FILES[@]} )); then
  echo "Found unlicensed files:
$(printf "%s\n" "${UNLICENSED_FILES[@]}")

Please include the following at the top of the file(s):
Copyright $(date +%Y) The StableHLO Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the \"License\");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an \"AS IS\" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."
  exit 1
fi

echo "Found no unlicensed files."
