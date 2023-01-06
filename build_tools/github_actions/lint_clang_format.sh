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

print_usage() {
  echo "Usage: $0 [-fd]"
  echo "    -f           Auto-fix clang-format issues."
  echo "    -b <branch>  Base branch name, default to origin/main."
}

FORMAT_MODE='validate'
BASE_BRANCH="$(git merge-base HEAD origin/main)"
while getopts 'fb:' flag; do
  case "${flag}" in
    f) FORMAT_MODE="fix" ;;
    b) BASE_BRANCH="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $# -ne 0 ]] ; then
  print_usage
  exit 1
fi

echo "Gathering changed files..."
CHANGED_FILES=$(git diff $BASE_BRANCH HEAD --name-only --diff-filter=d | grep '.*\.h\|.*\.cpp' | xargs)
if [[ -z "$CHANGED_FILES" ]]; then
  echo "No files to format."
  exit 0
fi

echo "Running clang-format [mode=$FORMAT_MODE]..."
echo "  Files: $CHANGED_FILES"
if [[ $FORMAT_MODE == 'fix' ]]; then
  clang-format --style=google -i $CHANGED_FILES
else
  clang-format --style=google --dry-run --Werror $CHANGED_FILES
fi
