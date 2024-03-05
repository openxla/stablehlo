# Copyright 2024 The StableHLO Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of starlark functions for use in BUILD files that are useful for StableHLO."""

def is_bzlmod_enabled():
    """Determine whether bzlmod mode is enabled."""

    # If bzlmod is enabled, then `str(Label(...))` returns a canonical label,
    # these start with `@@`.
    return str(Label("//:invalid")).startswith("@@")

def workspace_name():
    """Return the name of the workspace."""
    return "_main" if is_bzlmod_enabled() else "stablehlo"
