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

# This script replaces versions in the following lines in stablehlo/dialect/Version.h
#  static Version getCurrentVersion() { return Version(x, y, z); }
#  static Version getWeek4Version() { return Version(a, b, c); }
#  static Version getWeek12Version() { return Version(m, n, p); }
# getCurrentVersion() will be (x, y, z + 1)
# getWeek4Version() - The most recent git tag that was created at least 28 days ago.
# getWeek12Version() - The most recent git tag that was created at least 84 days ago.

set -o errexit
set -o nounset
set -o pipefail

script_dir="$(dirname "$(realpath "$0")")"
version_h="$script_dir/../stablehlo/dialect/Version.h"

fetch_current_version() {
  # getCurrentVersion() { Version(X, Y, Z); }
  ver_str=$(grep -A1 getCurrentVersion "$version_h" | grep -o 'Version(.*[0-9])')
  REGEX="Version\(([0-9]+), ([0-9]+), ([0-9]+)\)"
  if [[ $ver_str =~ $REGEX ]]; then
    curr_ver=("${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}")
  else
    echo "Error: Could not find current version string in $version_h" >&2
    exit 1
  fi
}
fetch_current_version

# calculate next current ver (patch ver + 1)
next_curr_z=$((curr_ver[2] + 1))

utc_time=$(date -u +%s)

week_4_tag=""
week_12_tag=""

while IFS= read -r line; do
    # split line CSV
    IFS=',' read -r tag_ts tag_v <<< "$line"
    ts_diff=$(( (utc_time - tag_ts) / 86400 ))

    if [ -z "$week_4_tag" ] && [ "$ts_diff" -ge 28 ]; then
        week_4_tag=$tag_v
    fi

    if [ -z "$week_12_tag" ] && [ "$ts_diff" -ge 84 ]; then
        week_12_tag=$tag_v
        break
    fi
done < <(git for-each-ref --sort=taggerdate --format '%(taggerdate:unix),%(refname:short)' refs/tags | tail -40 | tac)

if [ -z "$week_4_tag" ] || [ -z "$week_12_tag" ]; then
  echo "Error: WEEK_4 or WEEK_12 tag not found." >&2
  exit 1
fi

week_4_tag=$(echo "$week_4_tag" | sed -n 's/.*v\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\).*/\1, \2, \3/p')
week_12_tag=$(echo "$week_12_tag" | sed -n 's/.*v\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\).*/\1, \2, \3/p')

if [ -z "$week_4_tag" ] || [ -z "$week_12_tag" ]; then
  echo "Error: Unable to parse the WEEK_4 or WEEK_12 tag" >&2
  exit 1
fi

echo "Next Current Version: ${curr_ver[0]}, ${curr_ver[1]}, $next_curr_z" >&2
echo "WEEK_4 Version: $week_4_tag" >&2
echo "WEEK_12 Version: $week_12_tag" >&2

echo "Saving..." >&2

sed -i -E \
-e "s/(static Version getCurrentVersion\(\) \{ return Version\()([0-9]+), ([0-9]+), ([0-9]+)(\); \})/\1\2, \3, $next_curr_z\5/" \
-e "s/(static Version getWeek4Version\(\) \{ return Version\()([0-9]+), ([0-9]+), ([0-9]+)(\); \})/\1$week_4_tag\5/" \
-e "s/(static Version getWeek12Version\(\) \{ return Version\()([0-9]+), ([0-9]+), ([0-9]+)(\); \})/\1$week_12_tag\5/" \
"$version_h"

echo "Done" >&2
