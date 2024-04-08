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

set -o errexit
set -o nounset
set -o pipefail

if ! command -v buildifier &> /dev/null; then
  echo "Error: buildifier is not installed. Aborting."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly STABLEHLO_ROOT_DIR="${SCRIPT_DIR}/../.."

cd "$STABLEHLO_ROOT_DIR"

IFS=$'\n'
mapfile -t targets < <(find . -type f -name '*.bazel' -not -path './llvm*')

if ! buildifier --mode=check --lint=warn --warnings=all -r "${targets[@]}"; then
  echo "Error: buildifier failed. Please run: buildifier --mode=fix --lint=fix --warnings=all -r " "${targets[@]}"
  echo "You may need to install a specific version of buildifier (see the GitHub Actions workflow)."
  exit 1
fi
