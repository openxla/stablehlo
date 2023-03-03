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
  echo "Usage: $0 [-f]"
  echo "    -f           Auto-fix whitespace issues."
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

get_source_files() {
  git diff $BASE_BRANCH HEAD --name-only --diff-filter=d | grep '.*\.cpp$\|.*\.h$\|.*\.md$\|.*\.mlir$\|.*\.sh$\|.*\.td$\|.*\.txt$\|.*\.yml$\|.*\.yaml$' | xargs
}
echo "Checking whitespace:"
echo "  $(get_source_files)"

files_without_eof_newline() {
  [[ -z $(get_source_files) ]] || get_source_files | xargs -L1 bash -c 'test "$(tail -c 1 "$0")" && echo "$0"'
}

files_with_trailing_whitespace() {
  get_source_files | xargs grep -lP '[ \t]+$'
}

fix_files_without_eof_newline() {
  echo $1 | xargs  --no-run-if-empty sed -i -e '$a\'
}

fix_files_with_trailing_whitespace() {
  echo $1 | xargs --no-run-if-empty sed -i 's/[ \t]*$//'
}

EOF_NL=$(files_without_eof_newline)
TRAIL_WS=$(files_with_trailing_whitespace)

if [[ $FORMAT_MODE == 'fix' ]]; then
  echo "Fixing EOF newlines..."
  fix_files_without_eof_newline "$EOF_NL"
  echo "Fixing trailing whitespaces..."
  fix_files_with_trailing_whitespace "$TRAIL_WS"
else
  if [ ! -z "$EOF_NL$TRAIL_WS" ]; then
    echo "Missing newline at EOF:"
    echo $EOF_NL
    echo "Has trailing whitespace:"
    echo $TRAIL_WS
    echo
    echo "Auto-fix using:"
    echo "  $ ./build_tools/github_actions/lint_whitespace_checks.sh -f"
    exit 1
  else
    echo "No whitespace issues found."
  fi
fi
