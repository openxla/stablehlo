#!/usr/bin/env bash
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

# This runs the necessary step to build a Python wheel for StableHLO
# At the moment it only builds a Linux variant of Python wheel for the
# default python3 version present on the system (set via GitHub action typically)
# TODO: Incorporate cibuildwheel and build manylinux wheels

set -o errexit
set -o nounset
set -o pipefail

# Collect the root of the project directory relative to this file
PROJECT_DIR="$(realpath "$(dirname "$0")"/../../)"

# Set the source directory and destination directory for the distribution
SRC_DIR="${PROJECT_DIR}/stablehlo/integrations/python"
OUT_DIR="${1:-$SRC_DIR/dist}"

echo "Building python wheel. You will find it at ${OUT_DIR}"
python3 -m build -w --outdir "${OUT_DIR}" "${SRC_DIR}"

echo "Testing that the python wheel works correctly"
# Create a new virtual environment
python3 -m venv venv
# Activate the virtual environment
# shellcheck disable=SC1091
source venv/bin/activate
# Install the wheel
pip install --no-index --find-links="${OUT_DIR}" stablehlo
# Run the smoke test script
python "${SRC_DIR}/tests/smoketest.py"

