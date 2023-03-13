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

print_usage() {
  echo "Usage:"
  echo "$0 [-g][-o output_dir] <llvm_build_dir> <stablehlo_build_dir>"
  echo "    -g               Generate HTML report (default false)"
  echo "    -o <output_dir>  Set the output to generate report (default /tmp/ccov)"
}

OUTPUT_DIR="/tmp/ccov"
GENERATE_HTML=false
while getopts 'go:' flag; do
  case "${flag}" in
    g) GENERATE_HTML=true ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $# -ne 2 ]] ; then
  print_usage
  exit 1
fi

LLVM_BUILD_DIR="$1"
STABLEHLO_BUILD_DIR="$2"

# Check for lcov
if ! command -v lcov &> /dev/null
then
    echo "lcov could not be found"
    echo "$ sudo apt install lcov"
    exit
fi

# Configure StableHLO
cmake -GNinja \
  -B"$STABLEHLO_BUILD_DIR" \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fprofile-arcs -ftest-coverage -fcoverage-mapping" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -fprofile-instr-generate -fprofile-arcs -ftest-coverage -fcoverage-mapping"

# Build and Check StableHLO
(cd "$STABLEHLO_BUILD_DIR" && ninja check-stablehlo)

BUILD_TOOLS_DIR="$(dirname "$(readlink -f "$0")")"

REPORT_DATE="$(date +'%Y_%m_%d_%H-%M-%S')"
REPORT_DIR="$OUTPUT_DIR/ccov_$REPORT_DATE"
LCOV_DATA="$REPORT_DIR/cov.info"

mkdir -p $REPORT_DIR

lcov --directory "$STABLEHLO_BUILD_DIR"  \
     --base-directory "stablehlo" \
     --gcov-tool "$BUILD_TOOLS_DIR/llvm_gcov.sh" \
     --capture -o "$LCOV_DATA" \
     --no-external \
     --exclude "*.inc" \
     --exclude "llvm-project/*" \
     --quiet

# Capture code coverage output
if [[ $GENERATE_HTML == true ]]; then
  exec 5>&1
  CCOV_OUT=$(genhtml $LCOV_DATA \
      --output-directory $REPORT_DIR \
      --ignore-errors source \
      --show-details \
      --sort \
      --prefix $(pwd) | tee /dev/fd/5)
  echo "HTML report at:"
  echo "  $REPORT_DIR"
fi

echo "LCOV data at:"
echo "  $LCOV_DATA"
echo
echo "Summary:"
lcov -l "$LCOV_DATA"
