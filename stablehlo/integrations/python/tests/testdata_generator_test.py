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

import mlir.dialects.check as check_dialect
import mlir.dialects.stablehlo as stablehlo_dialect
import mlir.ir as ir
from mlir.stablehlo.testdata_generator.testdata_generator_lib import testdata_generator
import numpy as np


MODULE_STR = """
module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
"""

MODULE_STR_IN_TESTDATA_FORMAT_WITH_CHECK_OPS = """
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

MODULE_STR_IN_TESTDATA_FORMAT_WITH_CUSTOM_CALL_AS_CHECK_OPS = """
module {
  func.func @main() -> tensor<2xi1> {
    %cst = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
    %golden = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf32>
    %0 = stablehlo.add %cst, %cst_0 : tensor<2xf32>
    %1 = stablehlo.custom_call @check.eq(%0, %golden) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    return %1: tensor<2xi1>
  }
}
"""

MODULE_STR_IN_TESTDATA_FORMAT_WITH_SHARED_USE_OF_CONSTANT_OP = """
module {
  func.func @main() -> tensor<2xi1> {
    %cst = stablehlo.constant dense<0.0> : tensor<2xf32>
    %0 = stablehlo.add %cst, %cst : tensor<2xf32>
    %1 = stablehlo.custom_call @check.eq(%0, %cst) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    return %1 : tensor<2xi1>
  }
}
"""

MODULE_STR_IN_TESTDATA_FORMAT_MUTI_OP_UNDER_TEST = """
module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    %1 = stablehlo.add %0, %0 : tensor<2xf32>
    return %1: tensor<2xf32>
  }
}
"""

EXPECTED_TESTDATA_MODULE_STR_1 = """
module {
  func.func @main() -> tensor<2xf32> {
    %cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
    %cst_1 = stablehlo.constant dense<[4.000000e+00, 6.000000e+00]> : tensor<2xf32>
    %0 = stablehlo.add %cst, %cst_0 : tensor<2xf32>
    %1 = stablehlo.custom_call @check.eq(%cst_1, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %0 : tensor<2xf32>
  }
}
"""

EXPECTED_TESTDATA_MODULE_STR_2 = """
module {
  func.func @main() -> tensor<2xf32> {
    %cst = stablehlo.constant dense<[1.000000e+01, 2.000000e+01]> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<[3.000000e+01, 4.000000e+01]> : tensor<2xf32>
    %cst_1 = stablehlo.constant dense<[4.000000e+01, 6.000000e+01]> : tensor<2xf32>
    %0 = stablehlo.add %cst, %cst_0 : tensor<2xf32>
    %1 = stablehlo.custom_call @check.eq(%cst_1, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %0 : tensor<2xf32>
  }
}
"""

EXPECTED_TESTDATA_MODULE_STR_3 = """
module {
  func.func @main() -> tensor<2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
    %0 = stablehlo.add %cst, %cst : tensor<2xf32>
    %1 = stablehlo.custom_call @check.eq(%cst_0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %0 : tensor<2xf32>
  }
}
"""

EXPECTED_TESTDATA_MODULE_STR_4 = """
module {
  func.func @main() -> tensor<2xf32> {
    %cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
    %cst_1 = stablehlo.constant dense<[8.000000e+00, 1.200000e+01]> : tensor<2xf32>
    %0 = stablehlo.add %cst, %cst_0 : tensor<2xf32>
    %1 = stablehlo.add %0, %0 : tensor<2xf32>
    %2 = stablehlo.custom_call @check.eq(%cst_1, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %1 : tensor<2xf32>
  }
}
"""


def test_testdata_generator(
    module_str: str, args: list, expected_output_str: str
) -> None:
  """Parses the MLIR module string, registers dialects, runs testdata_generator,

  and compares the output with the expected output.
  """
  with ir.Context() as ctx:
    stablehlo_dialect.register_dialect(ctx)
    check_dialect.register_dialect(ctx)
    module = ir.Module.parse(module_str)
    result_module = testdata_generator(module, args)
    expected_output_module = ir.Module.parse(expected_output_str)
    if str(result_module) != str(expected_output_module):
      raise AssertionError(
          "Output mismatch:\n"
          f"Expected:\n{expected_output_module}\n"
          f"Got:\n{result_module}"
      )


# List of test cases as tuples: (input_module_str, args, expected_output_str)
test_cases = [
    (
        # Typical use-case.
        MODULE_STR,
        [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
        EXPECTED_TESTDATA_MODULE_STR_1,
    ),
    (
        # To test
        #  - Input programs already in testdata format
        #  - If no concrete inputs are provoded, they are derived from the embedded
        #    stablehlo.constant ops in the program.
        MODULE_STR_IN_TESTDATA_FORMAT_WITH_CHECK_OPS,
        [],
        EXPECTED_TESTDATA_MODULE_STR_1,
    ),
    (
        # To test
        #  - Input programs already in testdata format
        #  - If concrete inputs are provoded, the embedded stablehlo.constant ops,
        #    feeding as program inputs, will be ignored.
        MODULE_STR_IN_TESTDATA_FORMAT_WITH_CHECK_OPS,
        [
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([30.0, 40.0], dtype=np.float32),
        ],
        EXPECTED_TESTDATA_MODULE_STR_2,
    ),
    (
        # To test
        #  - Input programs already in testdata format
        #  - Usage of custom_call ops as check ops.
        MODULE_STR_IN_TESTDATA_FORMAT_WITH_CUSTOM_CALL_AS_CHECK_OPS,
        [],
        EXPECTED_TESTDATA_MODULE_STR_1,
    ),
    (
        # To test
        #  - Input programs already in testdata format
        #  - Proper identification of the constants feeding to program input and
        #    check ops.
        MODULE_STR_IN_TESTDATA_FORMAT_WITH_SHARED_USE_OF_CONSTANT_OP,
        [],
        EXPECTED_TESTDATA_MODULE_STR_3,
    ),
    (
        # To test handling of programs with multiple operations.
        MODULE_STR_IN_TESTDATA_FORMAT_MUTI_OP_UNDER_TEST,
        [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
        EXPECTED_TESTDATA_MODULE_STR_4,
    ),
]

for module_str, args, expected_output_str in test_cases:
  test_testdata_generator(module_str, args, expected_output_str)
