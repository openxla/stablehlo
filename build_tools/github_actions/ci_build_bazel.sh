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

set -o errexit
set -o nounset
set -o pipefail

if [[ $# -ne 0 && $# -ne 3 ]] ; then
  echo "Usage: $0 [<bazel-diff.jar> <start_commit> <end_commit>]"
  echo "   Builds and tests all bazel targets."
  echo "   If start_commit and end_commit are specified, uses bazel-diff to"
  echo "   determine the set of targets to build based on code changes."
  echo
  echo "Example: Build all affected targets between current branch and main"
  echo "  $ curl -Lo /tmp/bazel-diff.jar https://github.com/Tinder/bazel-diff/releases/latest/download/bazel-diff_deploy.jar"
  echo "  $ ci_build_bazel.sh /tmp/bazel-diff.jar main $(git rev-parse --abbrev-ref HEAD)"
  exit 1
fi

bazel-test-all() {
  # Build and Test StableHLO
  bazel build --lockfile_mode=error //... --config=asan --config=ubsan
  bazel test //... --config=asan --config=ubsan
}

bazel-test-diff() {
  set -e
  WORKSPACE_PATH=$(git rev-parse --show-toplevel)
  BAZEL_DIFF_JAR="$1"
  PREVIOUS_REV="$2" # Starting Revision SHA
  FINAL_REV="$3" # Final Revision SHA
  BAZEL_PATH="$(which bazel)"

  STARTING_HASHES_JSON="/tmp/starting_hashes.json"
  FINAL_HASHES_JSON="/tmp/final_hashes.json"
  IMPACTED_TARGETS_PATH="/tmp/impacted_targets.txt"
  FILTERED_TARGETS_PATH="/tmp/filtered_targets.txt"

  bazel-diff() {
    java -jar "$BAZEL_DIFF_JAR" "$@"
  }

  git -C "$WORKSPACE_PATH" checkout "$PREVIOUS_REV" --quiet

  echo "Generating Hashes for Revision '$PREVIOUS_REV'"
  bazel-diff generate-hashes -w "$WORKSPACE_PATH" -b "$BAZEL_PATH" $STARTING_HASHES_JSON

  UNCOMMITTED_CHANGES="$(git status -s)"
  if [[ -n "$UNCOMMITTED_CHANGES" ]]; then
    echo "[WARNING] Uncommitted changes found, likely build byproducts:"
    echo "$UNCOMMITTED_CHANGES"
  fi

  git reset --hard "$PREVIOUS_REV"
  git -C "$WORKSPACE_PATH" checkout "$FINAL_REV" --quiet

  echo "Generating Hashes for Revision '$FINAL_REV'"
  bazel-diff generate-hashes -w "$WORKSPACE_PATH" -b "$BAZEL_PATH" $FINAL_HASHES_JSON

  echo "Determining Impacted Targets"
  bazel-diff get-impacted-targets -sh $STARTING_HASHES_JSON -fh $FINAL_HASHES_JSON -o $IMPACTED_TARGETS_PATH
  echo ""

  impacted_targets=()
  IFS=$'\n' read -d '' -r -a impacted_targets < $IMPACTED_TARGETS_PATH || true
  formatted_impacted_targets=$(IFS=$'\n'; echo "${impacted_targets[*]}")
  if [[ -z "$formatted_impacted_targets" ]]; then
    echo "No impacted targets from change."
    exit 0
  fi

  NUM_IMPACTED=$(echo "$formatted_impacted_targets" | wc -l)
  echo "[$NUM_IMPACTED] Impacted Targets between $PREVIOUS_REV and $FINAL_REV:"
  echo "$formatted_impacted_targets"
  echo ""

  # Remove external and duplicate targets.
  sort "$IMPACTED_TARGETS_PATH" | uniq | grep -v '//external' > "$FILTERED_TARGETS_PATH" || true

  # Build and Test impacted targets
  if [[ -s "$FILTERED_TARGETS_PATH" ]]; then
    echo "Building and Testing Impacted (Non-External) Targets..."
    bazel build --target_pattern_file="$FILTERED_TARGETS_PATH"
    bazel test --target_pattern_file="$FILTERED_TARGETS_PATH"
  else
    echo "No non-external impacted targets to build and test."
  fi
}

# Run bazel build and test
if [[ $# -eq 0 ]] ; then
  bazel-test-all
else
  bazel-test-diff "$@"
fi
