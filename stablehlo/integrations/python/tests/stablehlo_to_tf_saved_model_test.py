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

import os
import tempfile
import mlir.dialects.stablehlo as stablehlo
import mlir.ir as ir
import numpy as np
from mlir.stablehlo.savedmodel.stablehlo_to_tf_saved_model import InputLocation, stablehlo_to_tf_saved_model
import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

# Convert a stablehlo program, expressing a nn.Linear layer with constant values
# for weight and bias, to saved model.

mlir_module_string = """
module @linearmodule attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {

  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[2,2]{0,1}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.dot_general %arg2, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %3 = stablehlo.add %1, %2 : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>\n
  }
}
"""

ctx = ir.Context()
stablehlo.register_dialect(ctx)
module = ir.Module.parse(mlir_module_string, ctx)

input_locations = [
    InputLocation.parameter(name='linear_layer.bias'),
    InputLocation.parameter(name='linear_layer.weight'),
    InputLocation.input_arg(position=0),
]
state_dict = {
    'linear_layer.weight': np.array(
        [[0.19075723, -0.13815854], [0.46516803, 0.12362058]], dtype='float32'
    ),
    'linear_layer.bias': np.array([-0.37076423, 0.03301], dtype='float32'),
}


saved_model_dir = tempfile.mkdtemp()
stablehlo_to_tf_saved_model(
    module,
    saved_model_dir=saved_model_dir,
    input_locations=input_locations,
    state_dict=state_dict,
)

saved_model = saved_model_utils.read_saved_model(saved_model_dir)
assert saved_model != None
print(f'StableHLO convertion to TF Saved Model seems to work!')
