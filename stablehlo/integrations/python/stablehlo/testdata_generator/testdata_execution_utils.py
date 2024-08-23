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

from typing import Sequence
from absl import logging
from mlir import ir
from mlir.dialects import stablehlo as stablehlo_dialect
import numpy as np


def run_stablehlo_interpreter(
    module: ir.Module, args: Sequence[np.ndarray]
) -> Sequence[np.ndarray]:
  """Evaluates a StableHLO module.

  Args:
    module: The MLIR module in StableHLO dialect.
    args: Input data for the module as a sequence of NumPy arrays.

  Returns:
    Sequence[np.ndarray]: Evaluated results from the interpreter as a sequence
    of NumPy arrays.
  """
  inputs = [ir.DenseElementsAttr.get(arg) for arg in args]
  results = stablehlo_dialect.eval_module(module, inputs)
  np_results = [np.array(result) for result in results]

  return np_results
