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
  echo "Usage: $0 [-f] <path/to/llvm/version>"
  echo "    -f           Auto-fix whitespace issues."
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

LLVM_VERSION_DIR=$1

LLVM_DIFF=$(sed -n '/LLVM_COMMIT = /p' WORKSPACE | sed 's/LLVM_COMMIT = //; s/\"//g' | diff $LLVM_VERSION_DIR -)

update_llvm_commit_and_sha256() {
  echo "Retrieving LLVM Commit..."
  export LLVM_COMMIT="$(cat $LLVM_VERSION_DIR)"
  echo "LLVM_COMMIT: $LLVM_COMMIT"
  echo "Calculating SHA256..."
  export LLVM_SHA256="$(curl -sL https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz | shasum -a 256 | sed 's/ //g; s/-//g')"
  echo "LLVM_SHA256: $LLVM_SHA256"

  sed -i '/^LLVM_COMMIT/s/"[^"]*"/"'$LLVM_COMMIT'"/g' WORKSPACE
  sed -i '/^LLVM_SHA256/s/"[^"]*"/"'$LLVM_SHA256'"/g' WORKSPACE
}

if [[ $FORMAT_MODE == 'fix' ]]; then
  echo "Updating LLVM Commit & SHA256..."
  update_llvm_commit_and_sha256 $LLVM_VERSION_DIR
else
  if [ ! -z "$LLVM_DIFF" ]; then
    echo "LLVM commit out of sync:"
    echo $LLVM_DIFF
    echo
    echo "Auto-fix using:"
    echo "  $ lint_llvm_commit.sh -f <path/to/llvm/version>"
    exit 1
  else
    echo "No llvm commit mismatches found."
  fi
fi
