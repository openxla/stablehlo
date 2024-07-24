# Copyright 2024 The StableHLO Authors.
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
"""Tests for CHECK Python APIs."""

# pylint: disable=wildcard-import,undefined-variable

from mlir import ir
from mlir.dialects import check as check_dialect
from mlir.dialects import stablehlo as stablehlo_dialect


def run(f):
  with ir.Context() as context:
    check_dialect.register_dialect(context)
    stablehlo_dialect.register_dialect(context)
    f()
  return f

@run
def test_parse():
  asm = """
    module {
      func.func @main() {
        %cst = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
        %cst_0 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
        %0 = stablehlo.add %cst, %cst_0 : tensor<2xf32>
        check.expect_eq_const %0, dense<[4.0, 6.0]> : tensor<2xf32>
        return
      }
    }
  """
  ir.Module.parse(asm)
