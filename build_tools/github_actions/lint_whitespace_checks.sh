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
  echo "    -f           Auto-fix whitespace issues."
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
mapfile -t CHANGED_FILES < <(git diff "$BASE_BRANCH" HEAD --name-only --diff-filter=d | grep -Ev '.*\.(bc|png|svg|ipynb)$')
if (( ${#CHANGED_FILES[@]} == 0 )); then
  echo "No files to check."
  exit 0
fi

echo "${CHANGED_FILES[@]}"

files_without_eof_newline() {
  # shellcheck disable=SC2016
  printf "%s\n" "${CHANGED_FILES[@]}" | xargs -L1 bash -c 'test "$(tail -c 1 "$0")" && echo "$0"'
}

files_with_trailing_whitespace() {
  printf "%s\n" "${CHANGED_FILES[@]}" | xargs -L1 grep -lP '[ \t]+$'
}

fix_files_without_eof_newline() {
  # shellcheck disable=SC2016,SC1003
  echo "$@" | xargs --no-run-if-empty sed -i -e '$a\'
}

fix_files_with_trailing_whitespace() {
  echo "$@" | xargs --no-run-if-empty sed -i 's/[ \t]*$//'
}

mapfile -t EOF_NL < <(files_without_eof_newline)
mapfile -t TRAIL_WS < <(files_with_trailing_whitespace)

if [[ $FORMAT_MODE == 'fix' ]]; then
  echo "Fixing EOF newlines..."
  fix_files_without_eof_newline "${EOF_NL[@]}"
  echo "Fixing trailing whitespaces..."
  fix_files_with_trailing_whitespace "${TRAIL_WS[@]}"
else
  if (( ${#EOF_NL[@]} + ${#TRAIL_WS[@]} )); then
    echo "Missing newline at EOF:"
    echo "${EOF_NL[@]}"
    echo "Has trailing whitespace:"
    echo "${TRAIL_WS[@]}"
    echo
    echo "Auto-fix using:"
    echo "  $ ./build_tools/github_actions/lint_whitespace_checks.sh -f"
    exit 1
  else
    echo "No whitespace issues found."
  fi
fi
