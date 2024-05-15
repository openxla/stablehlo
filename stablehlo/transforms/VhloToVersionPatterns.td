/* Copyright 2024 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/VhloOps.td"

def VHLO_GetDefaultConvPadding : NativeCodeCall<"getDefaultConvPadding($_builder, $0)">;

def DynamicConvUpgradeV1ToV2:
  Pat<(VHLO_DynamicConvOpV1 $lhs, $rhs, $d_padding, $window_strides, $padding, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      (VHLO_DynamicConvOpV2 $lhs, $rhs, $d_padding, $window_strides, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config)>;

def DynamicConvDowngradeV2ToV1:
  Pat<(VHLO_DynamicConvOpV2 $lhs, $rhs, $d_padding, $window_strides, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      (VHLO_DynamicConvOpV1 $lhs, $rhs, $d_padding, $window_strides, (VHLO_GetDefaultConvPadding $lhs), $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config)>;
