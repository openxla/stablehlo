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
  echo "    -s           Skip sha256 hash validation."
}

FORMAT_MODE='validate'
VALIDATE_SHA256='true'
while getopts 'fs' flag; do
  case "${flag}" in
    f) FORMAT_MODE="fix" ;;
    s) VALIDATE_SHA256="false" ;;
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

## Helper functions

# Commit validation functions
llvm_commit_from_version_txt() {
  cat $PATH_TO_LLVM_VERSION_TXT
}
llvm_commit_from_workspace() {
  sed -n '/LLVM_COMMIT = /p' $PATH_TO_WORKSPACE | sed 's/LLVM_COMMIT = //; s/\"//g'
}
llvm_commit_diff() {
  diff <(llvm_commit_from_version_txt) <(llvm_commit_from_workspace)
}

# SHA256 validation functions
llvm_sha256_from_workspace() {
  sed -n '/LLVM_SHA256 = /p' $PATH_TO_WORKSPACE  | sed 's/LLVM_SHA256 = //; s/\"//g'
}
llvm_sha256_from_archive() {
  LLVM_COMMIT=$(llvm_commit_from_workspace)
  HTTP_CODE=$(curl -sIL https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz -o /dev/null -w "%{http_code}")
  if [[ "$HTTP_CODE" == "404" ]]; then
    echo "Error 404 downloading LLVM at commit '$LLVM_COMMIT'."
    exit 1
  fi
  LLVM_SHA256="$(curl -sL https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz | shasum -a 256 | sed 's/ //g; s/-//g')"
  echo "$LLVM_SHA256"
}
llvm_sha256_diff() {
  diff <(llvm_sha256_from_workspace) <(llvm_sha256_from_archive)
}

# Fix functions
print_autofix() {
  echo "Auto-fix using:"
  echo "  $ lint_llvm_commit.sh -f <path/to/stablehlo/root>"
}

update_llvm_commit_and_sha256() {
  LLVM_COMMIT=$(llvm_commit_from_version_txt)
  LLVM_SHA256=$(llvm_sha256_from_archive)
  echo "Bumping commit to: $LLVM_COMMIT"
  sed -i '/^LLVM_COMMIT/s/"[^"]*"/"'$LLVM_COMMIT'"/g' $PATH_TO_WORKSPACE
  echo "Bumping sha256 to: $LLVM_SHA256"
  sed -i '/^LLVM_SHA256/s/"[^"]*"/"'$LLVM_SHA256'"/g' $PATH_TO_WORKSPACE
}

## Script body
if [[ $FORMAT_MODE == 'fix' ]]; then
  echo "Updating LLVM Commit & SHA256"
  update_llvm_commit_and_sha256
  exit 0
fi

echo "Validating LLVM commit hash..."
LLVM_COMMIT_DIFF=$(llvm_commit_diff)
if [[ ! -z "$LLVM_COMMIT_DIFF" ]]; then
  echo "Commit mismatch:"
  echo "$LLVM_COMMIT_DIFF"
  echo
  print_autofix
  exit 1
fi
echo "Commit hashes match."

if [[ "$VALIDATE_SHA256" == 'true' ]]; then
  echo "Validating LLVM SHA256 hash..."
  LLVM_SHA256_DIFF=$(llvm_sha256_diff)
  if [[ ! -z "$LLVM_SHA256_DIFF" ]]; then
    echo "...SHA256 mismatch:"
    echo "$LLVM_SHA256_DIFF"
    echo
    print_autofix
    exit 1
  fi
  echo "SHA256 hashes match."
fi
