#!/bin/bash
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

# This runs shellcheck on relevant files. shellcheck is a shell script analysis
# tool that can find bugs and subtle issues in shell scripts:
# https://www.shellcheck.net/

set -o errexit
set -o nounset
set -o pipefail

if ! command -v shellcheck &> /dev/null; then
  echo "Error: shellcheck is not installed. Aborting."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly STABLEHLO_ROOT_DIR="${SCRIPT_DIR}/../.."

cd "$STABLEHLO_ROOT_DIR"

IFS=$'\n'
mapfile -t targets < <(find . -type f -name '*.sh' -not -path './llvm*')

echo "Running shellcheck:"
shellcheck --version

if ! shellcheck --severity=style "${targets[@]}"; then
  echo "Error: shellcheck failed. Please run the following command to fix all issues:"
  echo "shellcheck --severity=style --format=diff" "${targets[@]}" "| git apply"
  echo "You may need to install a specific version of shellcheck (see above output)."
  exit 1
fi
