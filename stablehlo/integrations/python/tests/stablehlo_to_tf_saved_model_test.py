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
from mlir.stablehlo.savedmodel.stablehlo_to_tf_saved_model import InputLocation, stablehlo_to_tf_saved_model
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

# Convert a stablehlo program, expressing addition of an argument with constant
# values for weight and bias, to saved model.

MODULE_STRING = """
module @linearmodule attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {

  func.func @main(%bias: tensor<1xf32>, %weight: tensor<1xf32>, %arg0: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.add %arg0, %weight: tensor<1xf32>
    %1 = stablehlo.add %0, %bias : tensor<1xf32>
    return %1 : tensor<1xf32>\n
  }
}
"""

ctx = ir.Context()
stablehlo.register_dialect(ctx)
module = ir.Module.parse(MODULE_STRING, ctx)

input_locations = [
    InputLocation.parameter(name='linear_layer.bias'),
    InputLocation.parameter(name='linear_layer.weight'),
    InputLocation.input_arg(position=0),
]
state_dict = {
    'linear_layer.weight': np.array([1], dtype='float32'),
    'linear_layer.bias': np.array([2], dtype='float32'),
}


saved_model_dir = tempfile.mkdtemp()
stablehlo_version = stablehlo.get_current_version()
stablehlo_to_tf_saved_model(
    module,
    saved_model_dir=saved_model_dir,
    target_version=stablehlo_version,
    input_locations=input_locations,
    state_dict=state_dict,
)

restored_model = tf.saved_model.load(saved_model_dir)
restored_result = restored_model.f(tf.constant([3], tf.float32))
assert np.allclose(restored_result[0], tf.constant([6], tf.float32))
