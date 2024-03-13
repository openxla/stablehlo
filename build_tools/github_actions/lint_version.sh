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

# This lint validates that there are compatibility tests for the current
# version by extracting the current version vX.Y.Z from Version.h and
# checking if files named stablehlo_legalize_to_vhlo.X_Y_0.mlir and .mlir.bc
# exist.

set -o errexit
set -o nounset
set -o pipefail

## Setup VERSION variable as global:
VERSION_H="stablehlo/dialect/Version.h"
set_version_var() {
  # getCurrentVersion() { Version(0, X, Y); }
  VERSION_STR=$(cat $VERSION_H | grep getCurrentVersion -A1 | grep -o 'Version(.*[0-9])')
  REGEX="Version\(([0-9]+), ([0-9]+), ([0-9]+)\)"
  if [[ $VERSION_STR =~ $REGEX ]]; then
    VERSION=("${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}")
  else
    echo "Error: Could not find current version string in $VERSION_H" >&2
    exit 1
  fi
}
set_version_var

## Check if compatibility tests exist for version `X_Y_0`
COMPAT_TEST_BASE="stablehlo/tests/vhlo/stablehlo_legalize_to_vhlo"
TEST_VERSION="${VERSION[0]}_${VERSION[1]}_0"
VERSIONED_COMPAT_TEST="$COMPAT_TEST_BASE.$TEST_VERSION.mlir"
VERSIONED_COMPAT_TEST_BC="$COMPAT_TEST_BASE.$TEST_VERSION.mlir.bc"

show_help() {
  HELP_URL="https://github.com/openxla/stablehlo/blob/main/docs/vhlo.md#add-versioned-serialization-test"
  echo "For details on creating versioned tests for a new minor version of"
  echo "StableHLO, see the instructions on:"
  echo "$HELP_URL"
}

if ! test -e "$VERSIONED_COMPAT_TEST"; then
  echo "Error: Could not find compatibility test $VERSIONED_COMPAT_TEST"
  show_help
  exit 1
fi

if ! test -e "$VERSIONED_COMPAT_TEST_BC"; then
  echo "Error: Could not find compatibility bytecode test $VERSIONED_COMPAT_TEST_BC"
  show_help
  exit 1
fi
