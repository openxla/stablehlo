# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
#
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
# ==============================================================================
"""Tests for CHLO Python APIs."""

# pylint: disable=wildcard-import,undefined-variable

from mlir import ir
from mlir.dialects import chlo


def run(f):
  with ir.Context() as context:
    chlo.register_dialect(context)
    f()
  return f


@run
def test_comparison_direction_attr():
  attr = chlo.ComparisonDirectionAttr.get("EQ")
  assert attr is not None
  assert str(attr) == ("#chlo<comparison_direction EQ>")
  assert attr.value == "EQ"


@run
def test_comparison_type_attr():
  attr = chlo.ComparisonTypeAttr.get("FLOAT")
  assert attr is not None
  assert str(attr) == ("#chlo<comparison_type FLOAT>")
  assert attr.value == "FLOAT"
