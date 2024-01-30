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

# This runs buildifier on relevant files
# This runs markdownlint-cli with the specified files, using Docker.
# If passing the files as a glob, be sure to wrap in quotes. For more info,
# see https://github.com/igorshubovych/markdownlint-cli#globbing

set -o errexit
set -o nounset
set -o pipefail

readonly SCRIPT_DIR;
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly STABLEHLO_ROOT_DIR="${SCRIPT_DIR}/../.."

cd "$STABLEHLO_ROOT_DIR"

IFS=$'\n'
mapfile -t targets < <(find . -type f -name '*.sh' -not -path './llvm*' -printf '%P\0')

echo "Running shellcheck:"
shellcheck --version

if ! shellcheck --severity=info "${targets[@]}"; then
  echo "Error: shellcheck failed. Please run the following command to fix all issues:"
  echo "shellcheck --severity=info --format=diff" "${targets[@]}" "| git apply"
  echo "You may need to install a specific version of shellcheck (see above output)."
  exit 1
fi
