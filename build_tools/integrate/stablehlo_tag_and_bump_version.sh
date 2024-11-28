#!/bin/bash
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

exit_with_usage() {
  echo "Usage: $0 [-t <COMMIT_TO_TAG>]"
  echo "   -t  Specify a commit to tag, must be an integrated StableHLO commit"
  echo "       available on https://github.com/search?q=repo%3Aopenxla%2Fxla+integrate+stablehlo&type=commits"
  echo "       If not specifed, will only bump the 4w and 12w versions."
  exit 1
}

## Setup global variables:
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VERSION=""
VERSION_H="$SCRIPT_DIR/../../stablehlo/dialect/Version.h"
VERSION_CPP="$SCRIPT_DIR/../../stablehlo/dialect/Version.cpp"
setup_version_vars() {
  # getCurrentVersion() { Version(0, X, Y); }
  local VERSION_STR
  VERSION_STR=$(grep getCurrentVersion -A1 "$VERSION_H" | grep -o 'Version([0-9], .*)')
  local REGEX="Version\(([0-9]+), ([0-9]+), ([0-9]+)\)"
  if [[ $VERSION_STR =~ $REGEX ]]
  then
    VERSION=("${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}")
  else
    echo "Error: Could not find current version string in Version.h" >&2
    exit 1
  fi
}
setup_version_vars

## Tag old version
tag_integrated_version() {
  local COMMIT="$1"
  local RELEASE_TAG="$2"

  # Must be valid commit
  if ! git rev-parse "$COMMIT" > /dev/null; then
    echo "Could not find commit $COMMIT."
    echo "Sync with upstream and pull changes to local repo."
    exit 1
  fi

  # Ensure proper commit
  # NOTE: THIS COMMIT MUST BE INTEGRATED INTO OPENXLA/XLA BY STABLEHLO ONCALL
  # Valid commit hashes here:
  # https://github.com/search?q=repo%3Aopenxla%2Fxla+integrate+stablehlo&type=commits
  echo "Using commit:"
  git log --format=%B -n 1 "$COMMIT"
  echo
  read -p "Is this the correct commit? [y] " -n 1 -r
  echo; echo
  if [[ "$REPLY" =~ ^[Nn]$ ]]; then
    echo "Exiting..."
    exit 1
  fi

  echo "Creating tagged release $RELEASE_TAG at $COMMIT"
  echo "$ git tag -a $RELEASE_TAG $COMMIT -m \"StableHLO $RELEASE_TAG\""
  git tag -a "$RELEASE_TAG" "$COMMIT" -m "StableHLO $RELEASE_TAG"
  echo
  echo "Most recent tags:"
  git tag --sort=-version:refname | head -3
  echo
  echo "If this is incorrect, can undo using:"
  echo "$ git tag -d $RELEASE_TAG"
  echo
}

bump_patch_version() {
  ## Bump patch version in Version.h
  local VERSION_STR="Version(${VERSION[0]}, ${VERSION[1]}, ${VERSION[2]})"
  local NEW_VERSION_STR="Version(${VERSION[0]}, ${VERSION[1]}, $((VERSION[2]+1)))"
  echo "Bumping revision to: $NEW_VERSION_STR"
  echo "$ sed -i \"s/$VERSION_STR/$NEW_VERSION_STR/\" $VERSION_H"
  sed -i "s/$VERSION_STR/$NEW_VERSION_STR/" $VERSION_H
  echo
}

tag_and_bump() {
  local COMMIT="$1"
  local RELEASE="${VERSION[0]}.${VERSION[1]}.${VERSION[2]}"
  local RELEASE_TAG="v$RELEASE"
  local NEXT_RELEASE="${VERSION[0]}.${VERSION[1]}.$((VERSION[2]+1))"
  tag_integrated_version "$COMMIT" "$RELEASE_TAG"
  bump_patch_version

  ## Next steps
  echo "NEXT STEPS"
  echo "  Push tag to upstream using:"
  echo "  $ git push upstream $RELEASE_TAG"
  echo
  echo "  Commit and patch bump changes:"
  echo "  $ git add ."
  echo "  $ git commit -m \"Bump patch version after integrate $RELEASE -> $NEXT_RELEASE\""
}

## Update the 4w and 12w forward compatiblility versions
# This function:
# Replaces WEEK_4 and WEEK12 versions in stablehlo/dialect/Version.cpp
#    return Version(a, b, c);  // WEEK_4 ANCHOR: DO NOT MODIFY
#    return Version(m, n, p);  // WEEK_12 ANCHOR: DO NOT MODIFY
# WEEK_4 version - The most recent git tag that was created at least 28 days ago.
# WEEK_12 version - The most recent git tag that was created at least 84 days ago.

update_forward_compat_versions() {
  echo "Bumping 4w and 12w compatibility window values"

  local UTC_TIME
  UTC_TIME=$(date -u +%s)

  local WEEK_4_TAG=""
  local WEEK_12_TAG=""

  git fetch --tags upstream

  # Iterate over tags finding the proper tag
  while IFS= read -r line; do
      # split line CSV
      IFS=',' read -r tag_ts tag_v <<< "$line"
      ts_diff=$(( (UTC_TIME - tag_ts) / 86400 ))

      if [ -z "$WEEK_4_TAG" ] && [ "$ts_diff" -ge 28 ]; then
          WEEK_4_TAG=$tag_v
      fi

      if [ -z "$WEEK_12_TAG" ] && [ "$ts_diff" -ge 84 ]; then
          WEEK_12_TAG=$tag_v
          break
      fi
  done < <(git for-each-ref --sort=taggerdate --format '%(taggerdate:unix),%(refname:short)' refs/tags | tail -40 | tac)

  if [ -z "$WEEK_4_TAG" ] || [ -z "$WEEK_12_TAG" ]; then
    echo "Error: WEEK_4 or WEEK_12 tag not found." >&2
    exit 1
  fi

  WEEK_4_TAG=$(echo "$WEEK_4_TAG" | sed -n 's/.*v\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\).*/\1, \2, \3/p')
  WEEK_12_TAG=$(echo "$WEEK_12_TAG" | sed -n 's/.*v\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\).*/\1, \2, \3/p')

  if [ -z "$WEEK_4_TAG" ] || [ -z "$WEEK_12_TAG" ]; then
    echo "Error: Unable to parse the WEEK_4 or WEEK_12 tag" >&2
    exit 1
  fi

  echo "New WEEK_4 Version: $WEEK_4_TAG" >&2
  echo "New WEEK_12 Version: $WEEK_12_TAG" >&2

  sed -i -E \
    -e "s/(return Version\()([0-9]+), ([0-9]+), ([0-9]+)(\);\s+\/\/ WEEK_4 ANCHOR: DO NOT MODIFY)/\1$WEEK_4_TAG\5/" \
    -e "s/(return Version\()([0-9]+), ([0-9]+), ([0-9]+)(\);\s+\/\/ WEEK_12 ANCHOR: DO NOT MODIFY)/\1$WEEK_12_TAG\5/" \
    "$VERSION_CPP"


  # Checking Version.cpp values
  grep "WEEK_4 ANCHOR: DO NOT MODIFY" "$VERSION_CPP" >&2
  grep "WEEK_12 ANCHOR: DO NOT MODIFY" "$VERSION_CPP" >&2

  if [ "$(grep -c -E "return Version\($WEEK_4_TAG\);\s+\/\/ WEEK_4 ANCHOR: DO NOT MODIFY" "$VERSION_CPP")" -ne 1 ]; then
      echo "ERROR: WEEK_4 version is not correct" >&2
      exit 1
  fi

  if [ "$(grep -c -E "return Version\($WEEK_12_TAG\);\s+\/\/ WEEK_12 ANCHOR: DO NOT MODIFY" "$VERSION_CPP")" -ne 1 ]; then
      echo "ERROR: WEEK_12 version is not correct" >&2
      exit 1
  fi
}

# ------
# MAIN
# ------

set -o errexit  # Exit immediately if any command returns a non-zero exit status
set -o nounset  # Using uninitialized variables raises error and exits
set -o pipefail # Ensures the script detects errors in any part of a pipeline.

COMMIT_TO_TAG=""
while getopts 't:' flag; do
  case "${flag}" in
    t) COMMIT_TO_TAG="$OPTARG" ;;
    *) exit_with_usage ;;
  esac
done
shift $(( OPTIND - 1 ))

## Validation
if [ $# -ne 0 ]; then
  exit_with_usage
fi

# Must have upstream remote - This is needed to fetch and add tags
if ! git remote | grep upstream > /dev/null; then
  echo "Missing upstream remote. Use:"
  echo "$ git remote add upstream https://github.com/openxla/stablehlo.git"
  exit 1
fi

update_forward_compat_versions

if [ -n "$COMMIT_TO_TAG" ]; then
  tag_and_bump "$COMMIT_TO_TAG"
else
  echo "No commit to tag specified, only bumping 4w and 12w values."
fi
