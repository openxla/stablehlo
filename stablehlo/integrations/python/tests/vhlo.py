# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2023 The StableHLO Authors.
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
"""Tests for VHLO Python APIs."""

# pylint: disable=wildcard-import,undefined-variable

from mlir import ir
from mlir.dialects import vhlo


def run(f):
  with ir.Context() as context:
    vhlo.register_dialect(context)
    f()
  return f

@run
def test_parse():
  asm = """
    vhlo.func_v1 @main() -> () {
      "vhlo.return_v1"() : () -> ()
    } {
      arg_attrs = #vhlo.array_v1<[]>,
      res_attrs = #vhlo.array_v1<[]>,
      sym_visibility = #vhlo.string_v1<"public">
    }
  """
  ir.Module.parse(asm)
