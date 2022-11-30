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
"""Tests for StableHLO Python APIs."""

# pylint: disable=wildcard-import,undefined-variable

from mlir import ir
from mlir.dialects import stablehlo


def run(f):
  with ir.Context() as context:
    stablehlo.register_dialect(context)
    f()
  return f


@run
def test_channel_handle():
  attr = stablehlo.ChannelHandle.get(handle=1, type=2)
  assert attr is not None
  assert attr.handle == 1
  assert attr.channel_type == 2


@run
def test_comparison_direction_attr():
  attr = stablehlo.ComparisonDirectionAttr.get("EQ")
  assert attr is not None
  assert str(attr) == ("#stablehlo<comparison_direction EQ>")
  assert attr.value == "EQ"


@run
def test_comparison_type_attr():
  attr = stablehlo.ComparisonTypeAttr.get("FLOAT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<comparison_type FLOAT>")
  assert attr.value == "FLOAT"


@run
def test_conv_dimension_numbers():
  attr = stablehlo.ConvDimensionNumbers.get(
      input_batch_dimension=0,
      input_feature_dimension=1,
      input_spatial_dimensions=[2, 3, 4],
      kernel_input_feature_dimension=0,
      kernel_output_feature_dimension=1,
      kernel_spatial_dimensions=[2, 3],
      output_batch_dimension=0,
      output_feature_dimension=1,
      output_spatial_dimensions=[2, 3])
  assert str(attr) == ("#stablehlo.conv<[b, f, 0, 1, 2]x[i, o, 0, 1]->"
                       "[b, f, 0, 1]>")
  assert attr is not None
  assert attr.input_batch_dimension == 0
  assert attr.input_feature_dimension == 1
  assert attr.input_spatial_dimensions == [2, 3, 4]
  assert attr.kernel_input_feature_dimension == 0
  assert attr.kernel_output_feature_dimension == 1
  assert attr.kernel_spatial_dimensions == [2, 3]
  assert attr.output_batch_dimension == 0
  assert attr.output_feature_dimension == 1
  assert attr.output_spatial_dimensions == [2, 3]


@run
def test_dot_dimension_numbers():
  attr = stablehlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=[0, 1],
      rhs_batching_dimensions=[2, 3],
      lhs_contracting_dimensions=[4, 5],
      rhs_contracting_dimensions=[6, 7])
  assert attr is not None
  assert str(attr) == ("#stablehlo.dot<lhs_batching_dimensions = [0, 1], "
                       "rhs_batching_dimensions = [2, 3], "
                       "lhs_contracting_dimensions = [4, 5], "
                       "rhs_contracting_dimensions = [6, 7]>")
  assert attr.lhs_batching_dimensions == [0, 1]
  assert attr.rhs_batching_dimensions == [2, 3]
  assert attr.lhs_contracting_dimensions == [4, 5]
  assert attr.rhs_contracting_dimensions == [6, 7]


@run
def test_fft_type_attr():
  attr = stablehlo.FftTypeAttr.get("FFT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<fft_type FFT>")
  assert attr.value == "FFT"


@run
def test_gather_dimension_numbers():
  attr = stablehlo.GatherDimensionNumbers.get(
      offset_dims=[1, 2],
      collapsed_slice_dims=[3, 4, 5],
      start_index_map=[6],
      index_vector_dim=7)
  assert attr is not None
  assert str(attr) == ("#stablehlo.gather<offset_dims = [1, 2], "
                       "collapsed_slice_dims = [3, 4, 5], "
                       "start_index_map = [6], "
                       "index_vector_dim = 7>")
  assert attr.offset_dims == [1, 2]
  assert attr.collapsed_slice_dims == [3, 4, 5]
  assert attr.start_index_map == [6]
  assert attr.index_vector_dim == 7


@run
def test_output_operand_alias():
  attr = stablehlo.OutputOperandAlias.get(
      output_tuple_indices=[0],
      operand_index=0,
      operand_tuple_indices=[1])
  assert attr is not None
  assert str(attr) == ("#stablehlo.output_operand_alias<output_tuple_indices = [0], "
                       "operand_index = 0, "
                       "operand_tuple_indices = [1]>")
  assert attr.output_tuple_indices == [0]
  assert attr.operand_index == 0
  assert attr.operand_tuple_indices == [1]


@run
def test_precision_attr():
  attr = stablehlo.PrecisionAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<precision DEFAULT>")
  assert attr.value == "DEFAULT"


@run
def test_rng_algorithm_attr():
  attr = stablehlo.RngAlgorithmAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<rng_algorithm DEFAULT>")
  assert attr.value == "DEFAULT"


@run
def test_rng_distribution_attr():
  attr = stablehlo.RngDistributionAttr.get("UNIFORM")
  assert attr is not None
  assert str(attr) == ("#stablehlo<rng_distribution UNIFORM>")
  assert attr.value == "UNIFORM"


@run
def test_scatter_dimension_numbers():
  attr = stablehlo.ScatterDimensionNumbers.get(
      update_window_dims=[1, 2, 3],
      inserted_window_dims=[4, 5],
      scattered_dims_to_operand_dims=[6, 7],
      index_vector_dim=8)
  assert attr is not None
  assert str(attr) == ("#stablehlo.scatter<update_window_dims = [1, 2, 3], "
                       "inserted_window_dims = [4, 5], "
                       "scatter_dims_to_operand_dims = [6, 7], "
                       "index_vector_dim = 8>")
  assert attr.update_window_dims == [1, 2, 3]
  assert attr.inserted_window_dims == [4, 5]
  assert attr.scattered_dims_to_operand_dims == [6, 7]
  assert attr.index_vector_dim == 8


@run
def test_transpose_attr():
  attr = stablehlo.TransposeAttr.get("TRANSPOSE")
  assert attr is not None
  assert str(attr) == ("#stablehlo<transpose TRANSPOSE>")
  assert attr.value == "TRANSPOSE"


@run
def test_token_type():
  type = stablehlo.TokenType.get()
  assert type is not None
  assert str(type) == "!stablehlo.token"


@run
def test_type_extensions():
  dyn_size = ir.ShapedType.get_dynamic_size()
  attr = stablehlo.TypeExtensions.get(bounds=[128, dyn_size])
  assert attr is not None
  assert attr.bounds == [128, dyn_size]
