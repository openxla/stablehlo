#!/bin/bash
# Copyright 2022 The StableHLO Authors.
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

exit_with_usage() {
  echo "Usage: $0 [-fb]"
  echo "    -f           Auto-fix clang-format issues."
  echo "    -b <branch>  Base branch name, defaults to main."
  exit 1
}

FORMAT_MODE='validate'
BASE_BRANCH=main
while getopts 'fb:' flag; do
  case "${flag}" in
    f) FORMAT_MODE="fix" ;;
    b) BASE_BRANCH="$OPTARG" ;;
    *) exit_with_usage ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $# -ne 0 ]] ; then
  exit_with_usage
fi

echo "Gathering changed files..."
mapfile -t CHANGED_FILES < <(git diff "$BASE_BRANCH" HEAD --name-only --diff-filter=d | grep '.*\.h\|.*\.cpp')
if (( ${#CHANGED_FILES[@]} == 0 )); then
  echo "No files to format."
  exit 0
fi

echo "Running clang-format [mode=$FORMAT_MODE]..."
echo "  Files: " "${CHANGED_FILES[@]}"
if [[ $FORMAT_MODE == 'fix' ]]; then
  clang-format -i "${CHANGED_FILES[@]}"
else
  clang-format --dry-run --Werror "${CHANGED_FILES[@]}" || {
    echo "Please format your code before pushing:"
    echo "  ./build_tools/github_actions/lint_clang_format.sh -f"
    echo "Make sure you are using the right major version of clang-format:"
    echo "  $(clang-format --version)"
    exit 1
  }
fi
