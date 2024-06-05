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

# This script builds the documentation and optionally checks that the
# documentation is up to date (building the documentation led to no changes).

set -o errexit
set -o nounset
set -o pipefail

exit_with_usage() {
  echo "Usage: $0 [-c]"
  echo "    -c Check if the docs are up to date."
  exit 1
}

if (( $# > 1 )); then
  exit_with_usage
fi

CHECK=
if (( $# == 1)) && [[ $1 == -c ]]; then
  CHECK=true
  shift
fi

if (( $# != 0 )); then
  exit_with_usage
fi

declare -A targets
targets[":stablehlo_pass_inc_gen_filegroup"]="bazel-bin/stablehlo/transforms/stablehlo_passes.md"
targets[":interpreter_pass_inc_gen_filegroup"]="bazel-bin/stablehlo/reference/interpreter_passes.md"
targets[":linalg_pass_inc_gen_filegroup"]="bazel-bin/stablehlo/conversions/linalg/transforms/stablehlo_linalg_passes.md"
targets[":tosa_pass_inc_gen_filegroup"]="bazel-bin/stablehlo/conversions/tosa/transforms/stablehlo_tosa_passes.md"

bazel build "${!targets[@]}"

cp "${targets[@]}" docs/generated

DOC_DIFF="$(git diff)"
[[ "$CHECK" ]] && [[ "$DOC_DIFF" ]] && {
  echo "$DOC_DIFF"
  echo
  echo "Generated pass documentation is out of date (see diff above)."
  echo "Re-generate the documentation before pushing using:"
  echo "  ./build_tools/github_actions/ci_build_docs.sh"
  exit 1
} || exit 0
