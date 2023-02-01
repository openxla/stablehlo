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
  echo "Usage: $0 [-f] <path/to/stablehlo/root>"
  echo "    -f           Auto-fix LLVM commit mismatch."
}

FORMAT_MODE='validate'
while getopts 'f' flag; do
  case "${flag}" in
    f) FORMAT_MODE="fix" ;;
    *) print_usage
       exit 1 ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $# -ne 1 ]] ; then
  print_usage
  exit 1
fi

PATH_TO_STABLEHLO_ROOT="$1"
PATH_TO_LLVM_VERSION_TXT="$PATH_TO_STABLEHLO_ROOT/build_tools/llvm_version.txt"
PATH_TO_WORKSPACE="$PATH_TO_STABLEHLO_ROOT/WORKSPACE.bazel"

retrieve_llvm_commit() {
  echo "Retrieving LLVM Commit..."
  export LLVM_COMMIT="$(<$PATH_TO_LLVM_VERSION_TXT)"
  echo "LLVM_COMMIT: $LLVM_COMMIT"
}

calculate_llvm_commit_sha256() {
  echo "Calculating SHA256..."
  if [[ $(curl -sL https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz | tr '\0' '\n') != "404: Not Found" ]]; then
    export LLVM_SHA256="$(curl -sL https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz | shasum -a 256 | sed 's/ //g; s/-//g')"
    echo "LLVM_SHA256: $LLVM_SHA256"
  else
    echo "LLVM_COMMIT: $LLVM_COMMIT in $PATH_TO_LLVM_VERSION_TXT not found."
    exit 1
  fi
}

calculate_diff() {
  export LLVM_DIFF=$(sed -n '/LLVM_COMMIT = /p' $PATH_TO_WORKSPACE | sed 's/LLVM_COMMIT = //; s/\"//g' | diff $PATH_TO_LLVM_VERSION_TXT -)
  export LLVM_SHA256_DIFF=$(sed -n '/LLVM_SHA256 = /p' $PATH_TO_WORKSPACE  | sed 's/LLVM_SHA256 = //; s/\"//g' | diff <(echo $LLVM_SHA256) -)
}

update_llvm_commit_and_sha256() {
  sed -i '/^LLVM_COMMIT/s/"[^"]*"/"'$LLVM_COMMIT'"/g' $PATH_TO_WORKSPACE
  sed -i '/^LLVM_SHA256/s/"[^"]*"/"'$LLVM_SHA256'"/g' $PATH_TO_WORKSPACE
}

retrieve_llvm_commit
calculate_llvm_commit_sha256
calculate_diff

if [[ $FORMAT_MODE == 'fix' ]]; then
  echo "Updating LLVM Commit & SHA256..."
  update_llvm_commit_and_sha256
  echo "Done."
else
  if [ ! -z "$LLVM_DIFF" ]; then
    echo "LLVM commit out of sync:"
    echo $LLVM_DIFF
    echo
    echo "Auto-fix using:"
    echo "  $ lint_llvm_commit.sh -f <path/to/stablehlo/root>"
    exit 1
  else
    echo "No llvm commit mismatches found."
    if [ ! -z "$LLVM_SHA256_DIFF" ]; then
      echo "LLVM SHA256 out of sync:"
      echo $LLVM_SHA256_DIFF
      echo $(sed -n '/LLVM_SHA256 = /p' $PATH_TO_WORKSPACE | sed 's/LLVM_SHA256 = //; s/\"//g')
      echo "Auto-fix using:"
      echo "  $ lint_llvm_commit.sh -f <path/to/stablehlo/root>"
      exit 1
    else
      echo "No sha256 mismatches found."
    fi
  fi
fi
