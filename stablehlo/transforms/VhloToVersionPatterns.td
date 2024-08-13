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

def VHLO_GetEmptyDims : NativeCodeCall<"getEmptyI64Tensor($_builder)">;

def VHLO_EmptyDims : AttrConstraint<CPred<"isEmptyTensor($_self)">, "Empty dims">;

def VHLO_NoneType : AttrConstraint<CPred<"isNoneType($_self)">, "None type">;

def VHLO_GetNoneType : NativeCodeCall<"getNoneType($_builder)">;

def VHLO_GetDefaultConvPadding : NativeCodeCall<"getDefaultConvPadding($_builder, $0)">;

def VHLO_IsSingleOperand :  Constraint<CPred<"$0.size() == 1">, "Single operand">;

def VHLO_GetFirstOperand : NativeCodeCall<"$0.front()">;

def VHLO_WrapInVector : NativeCodeCall<"{$0}">;

def DynamicConvUpgradeV1ToV2:
  Pat<(VHLO_DynamicConvOpV1 $lhs, $rhs, $d_padding, $window_strides, $padding, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      (VHLO_DynamicConvOpV2 $lhs, $rhs, $d_padding, $window_strides, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config)>;

def DynamicConvDowngradeV2ToV1:
  Pat<(VHLO_DynamicConvOpV2 $lhs, $rhs, $d_padding, $window_strides, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      (VHLO_DynamicConvOpV1 $lhs, $rhs, $d_padding, $window_strides, (VHLO_GetDefaultConvPadding $lhs), $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config)>;

def GatherOpUpgradeV1ToV2:
  Pat<(VHLO_GatherOpV1 $operand, $start_indices, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted),
      (VHLO_GatherOpV2 $operand, $start_indices, $offset_dims, $collapsed_slice_dims, (VHLO_GetEmptyDims), (VHLO_GetEmptyDims), $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted)>;

def GatherOpDowngradeV2ToV1 :
  Pat<(VHLO_GatherOpV2 $operand, $start_indices, $offset_dims, $collapsed_slice_dims, VHLO_EmptyDims:$operand_batching_dims, VHLO_EmptyDims:$start_indices_batching_dims, $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted),
      (VHLO_GatherOpV1 $operand, $start_indices, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted)>;

def DynamicGatherOpUpgradeV1ToV2:
  Pat<(VHLO_DynamicGatherOpV1 $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $indices_are_sorted),
      (VHLO_DynamicGatherOpV2 $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, (VHLO_GetEmptyDims), (VHLO_GetEmptyDims), $start_index_map, $index_vector_dim, $indices_are_sorted)>;

def DynamicGatherOpDowngradeV2ToV1 :
  Pat<(VHLO_DynamicGatherOpV2 $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, VHLO_EmptyDims:$operand_batching_dims, VHLO_EmptyDims:$start_indices_batching_dims, $start_index_map, $index_vector_dim, $indices_are_sorted),
      (VHLO_DynamicGatherOpV1 $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $indices_are_sorted)>;

def AllGatherOpUpgradeV1ToV2 :
  Pat<(VHLO_AllGatherOpV1 $operand, $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids),
      (VHLO_AllGatherOpV2 (VHLO_WrapInVector $operand), $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids)>;

def AllGatherOpDowngradeV2ToV1 :
  Pat<(VHLO_AllGatherOpV2 $operand, $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids),
      (VHLO_AllGatherOpV1 (VHLO_GetFirstOperand $operand), $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids),
      [(VHLO_IsSingleOperand $operand)]>;

def AllToAllOpUpgradeV1ToV2 :
  Pat<(VHLO_AllToAllOpV1 $operand, $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id),
      (VHLO_AllToAllOpV2 (VHLO_WrapInVector $operand), $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id)>;

def AllToAllOpDowngradeV2ToV1 :
  Pat<(VHLO_AllToAllOpV2 $operand, $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id),
      (VHLO_AllToAllOpV1 (VHLO_GetFirstOperand $operand), $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id),
      [(VHLO_IsSingleOperand $operand)]>;

def DotGeneralOpDowngradeV2ToV1 :
  Pat<(VHLO_DotGeneralOpV2 $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config,
         VHLO_NoneType:$lhs_precision_type, VHLO_NoneType:$rhs_precision_type, VHLO_NoneType:$accumulation_type, VHLO_NoneType:$lhs_component_count, VHLO_NoneType:$rhs_component_count, VHLO_NoneType:$num_primitive_operations, VHLO_NoneType:$allow_imprecise_accumulation),
      (VHLO_DotGeneralOpV1 $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config)>;

def DotGeneralOpUpradeV1ToV2 :
  Pat<(VHLO_DotGeneralOpV1 $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config),
      (VHLO_DotGeneralOpV2 $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config,
         (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType))>;
