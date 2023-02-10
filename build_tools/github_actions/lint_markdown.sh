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

# This runs markdownlint-cli with the specified files, using Docker.
# If passing the files as a glob, be sure to wrap in quotes. For more info,
# see https://github.com/igorshubovych/markdownlint-cli#globbing

if [[ $# -gt 2 ]] ; then
  echo "Usage: $0 [-f] <files|directories|globs>"
  echo "  -f  Autofix markdown issues."
  echo " "
  echo "All file/directory/glob paths must be relative to the repo root."
  echo "Glob patterns must be wrapped in quotes."
  echo "To lint all .md files, run this command:"
  echo "    bash lint_markdown.sh \"./**/*.md\""
  exit 1
fi

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly STABLEHLO_ROOT_DIR="${SCRIPT_DIR}/../.."

# These must be relative to the repo root because that's the context
# in which the Docker container will mount and find all files
readonly CONFIG=".markdownlint.yaml"

# Verify Docker is available
if ! command -v docker &> /dev/null
then
    echo "Error: You must install Docker."
    exit
fi

# Run markdownlint-cli in Docker to avoid node versioning issues
docker run -v $STABLEHLO_ROOT_DIR:/workdir \
    ghcr.io/igorshubovych/markdownlint-cli:v0.32.2 \
    --config $CONFIG "$@"
