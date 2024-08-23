# Copyright 2024 The StableHLO Authors. All Rights Reserved.
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
"""Testdata Generator Utils."""

from typing import Sequence

from absl import logging
from mlir import ir
from mlir.stablehlo.testdata_generator import testdata_execution_utils
from mlir.stablehlo.testdata_generator import testdata_processor
import numpy as np


def testdata_generator(
    module: ir.Module, args: Sequence[np.ndarray] = []
) -> ir.Module:
  """Generates test data for a StableHLO module.

  This function takes a StableHLO module and optional input arguments, processes
  the module to
  extract relevant information, executes the module to obtain golden results,
  and then converts
  the module, inputs, and golden results into a standardized test data format.

  Args:
    module: The StableHLO module to generate test data for.
    args: (Optional) A sequence of NumPy arrays representing input values for
      the module. If not provided, the function will attempt to extract input
      values from the module itself.

  Returns:
    An MLIR module in the test data format, containing the original module,
    inputs, and golden results.

  Example:
    Input (module_str):
    ```
    module {
      func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) ->
      tensor<2xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        return %0 : tensor<2xf32>
      }
    }
    ```

    Input (args):
    ```
    [
      np.array([1.0, 2.0], dtype=np.float32),
      np.array([3.0, 4.0], dtype=np.float32)
    ]
    ```

    Output (module_output_str):
    ```
    module {
      func.func @main() -> tensor<i1> {
        %cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> :
        tensor<2xf32>
        %cst_0 = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> :
        tensor<2xf32>
        %cst_1 = stablehlo.constant dense<[4.000000e+00, 6.000000e+00]> :
        tensor<2xf32>
        %0 = stablehlo.add %cst, %cst_0 : tensor<2xf32>
        %1 = stablehlo.custom_call @check.eq(%cst_1, %0) : (tensor<2xf32>,
        tensor<2xf32>) -> tensor<i1>
        return %1 : tensor<i1>
      }
    }
    ```
  """
  module, inputs = testdata_processor.preprocess_input_module(module)
  if args:
    inputs = args
  logging.info(
      f"\t[testdata-generator] Processed module and inputs: {module}, {inputs}"
  )

  golden_results = testdata_execution_utils.run_stablehlo_interpreter(
      module, inputs
  )
  logging.info(f"\t[testdata-generator] Golden results: {golden_results}")

  module_output = testdata_processor.to_testdata_format(
      module, inputs, golden_results
  )

  return module_output
