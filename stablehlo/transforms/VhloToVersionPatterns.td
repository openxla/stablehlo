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
include "mlir/IR/CommonAttrConstraints.td"
include "stablehlo/dialect/VhloEnums.td"
include "stablehlo/dialect/VhloAttrs.td"

def VHLO_GetEmptyTensor : NativeCodeCall<"getEmptyI64Tensor($_builder)">;

def VHLO_EmptyTensor : AttrConstraint<CPred<"isEmptyTensor($_self)">, "Empty dims">;

def VHLO_NoneType : AttrConstraint<CPred<"isNoneType($_self)">, "None type">;

def VHLO_GetNoneType : NativeCodeCall<"getNoneType($_builder)">;

def VHLO_GetDefaultConvPadding : NativeCodeCall<"getDefaultConvPadding($_builder, $0)">;

def VHLO_IsSingleOperand :  Constraint<CPred<"$0.size() == 1">, "Single operand">;

def VHLO_GetFirstOperand : NativeCodeCall<"$0.front()">;

def VHLO_WrapInVector : NativeCodeCall<"{$0}">;

def VHLO_GetDefaultResultAccuracyAttr : NativeCodeCall<"getDefaultResultAccuracy($_builder)">;

def VHLO_DefaultResultAccuracy : AttrConstraint<CPred<"isDefaultResultAccuracy($_self)">, "Default result accuracy">;

def VHLO_CopyDiscardableAttrs: NativeCodeCallVoid<"copyDiscardableAttrs($0, $1)">;

def DynamicConvUpgradeV1ToV2:
  Pat<(VHLO_DynamicConvOpV1:$src $lhs, $rhs, $d_padding, $window_strides, $padding, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      (VHLO_DynamicConvOpV2:$dst $lhs, $rhs, $d_padding, $window_strides, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def DynamicConvDowngradeV2ToV1:
  Pat<(VHLO_DynamicConvOpV2:$src $lhs, $rhs, $d_padding, $window_strides, $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      (VHLO_DynamicConvOpV1:$dst $lhs, $rhs, $d_padding, $window_strides, (VHLO_GetDefaultConvPadding $lhs), $lhs_dilation, $rhs_dilation, $window_reversal, $input_batch_dimension, $input_feature_dimension, $input_spatial_dimensions, $kernel_input_feature_dimension, $kernel_output_feature_dimension, $kernel_spatial_dimensions, $output_batch_dimension, $output_feature_dimension, $output_spatial_dimensions, $feature_group_count, $batch_group_count, $precision_config),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def GatherOpUpgradeV1ToV2:
  Pat<(VHLO_GatherOpV1:$src $operand, $start_indices, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted),
      (VHLO_GatherOpV2:$dst $operand, $start_indices, $offset_dims, $collapsed_slice_dims, (VHLO_GetEmptyTensor), (VHLO_GetEmptyTensor), $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def GatherOpDowngradeV2ToV1 :
  Pat<(VHLO_GatherOpV2:$src $operand, $start_indices, $offset_dims, $collapsed_slice_dims, VHLO_EmptyTensor:$operand_batching_dims, VHLO_EmptyTensor:$start_indices_batching_dims, $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted),
      (VHLO_GatherOpV1:$dst $operand, $start_indices, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $slice_sizes, $indices_are_sorted),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def DynamicGatherOpUpgradeV1ToV2:
  Pat<(VHLO_DynamicGatherOpV1:$src $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $indices_are_sorted),
      (VHLO_DynamicGatherOpV2:$dst $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, (VHLO_GetEmptyTensor), (VHLO_GetEmptyTensor), $start_index_map, $index_vector_dim, $indices_are_sorted),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def DynamicGatherOpDowngradeV2ToV1 :
  Pat<(VHLO_DynamicGatherOpV2:$src $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, VHLO_EmptyTensor:$operand_batching_dims, VHLO_EmptyTensor:$start_indices_batching_dims, $start_index_map, $index_vector_dim, $indices_are_sorted),
      (VHLO_DynamicGatherOpV1:$dst $operand, $start_indices, $slice_sizes, $offset_dims, $collapsed_slice_dims, $start_index_map, $index_vector_dim, $indices_are_sorted),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def AllGatherOpUpgradeV1ToV2 :
  Pat<(VHLO_AllGatherOpV1:$src $operand, $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids),
      (VHLO_AllGatherOpV2:$dst (VHLO_WrapInVector $operand), $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids),
      [], [(VHLO_CopyDiscardableAttrs $src, (VHLO_GetFirstOperand $dst))]>;

def AllGatherOpDowngradeV2ToV1 :
  Pat<(VHLO_AllGatherOpV2:$src $operand, $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids),
      (VHLO_AllGatherOpV1:$dst (VHLO_GetFirstOperand $operand), $all_gather_dim, $replica_groups, $channel_id, $use_global_device_ids),
      [(VHLO_IsSingleOperand $operand)], [(VHLO_CopyDiscardableAttrs (VHLO_GetFirstOperand $src), $dst)]>;

def AllToAllOpUpgradeV1ToV2 :
  Pat<(VHLO_AllToAllOpV1:$src $operand, $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id),
      (VHLO_AllToAllOpV2:$dst (VHLO_WrapInVector $operand), $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id),
      [], [(VHLO_CopyDiscardableAttrs $src, (VHLO_GetFirstOperand $dst))]>;

def AllToAllOpDowngradeV2ToV1 :
  Pat<(VHLO_AllToAllOpV2:$src $operand, $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id),
      (VHLO_AllToAllOpV1:$dst (VHLO_GetFirstOperand $operand), $split_dimension, $concat_dimension, $split_count, $replica_groups, $channel_id),
      [(VHLO_IsSingleOperand $operand)], [(VHLO_CopyDiscardableAttrs (VHLO_GetFirstOperand $src), $dst)]>;

def DotGeneralOpDowngradeV2ToV1 :
  Pat<(VHLO_DotGeneralOpV2:$src $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config,
         VHLO_NoneType:$lhs_precision_type, VHLO_NoneType:$rhs_precision_type, VHLO_NoneType:$accumulation_type, VHLO_NoneType:$lhs_component_count, VHLO_NoneType:$rhs_component_count, VHLO_NoneType:$num_primitive_operations, VHLO_NoneType:$allow_imprecise_accumulation),
      (VHLO_DotGeneralOpV1:$dst $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def DotGeneralOpUpradeV1ToV2 :
  Pat<(VHLO_DotGeneralOpV1:$src $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config),
      (VHLO_DotGeneralOpV2:$dst $lhs, $rhs, $lhs_batching_dimensions, $rhs_batching_dimensions, $lhs_contracting_dimensions, $rhs_contracting_dimensions, $precision_config,
         (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType), (VHLO_GetNoneType)),
         [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def SendOpDowngradeV2ToV1 :
  Pat<(VHLO_SendOpV2:$src $operand, $token, $channel_id, $channel_type, $is_host_transfer, VHLO_EmptyTensor:$source_target_pairs),
      (VHLO_SendOpV1:$dst $operand, $token, $channel_id, $channel_type, $is_host_transfer),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def SendOpUpgradeV1ToV2 :
  Pat<(VHLO_SendOpV1:$src $operand, $token, $channel_id, $channel_type, $is_host_transfer),
      (VHLO_SendOpV2:$dst $operand, $token, $channel_id, $channel_type, $is_host_transfer, (VHLO_GetEmptyTensor)),
      [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;

def RecvOpDowngradeV2ToV1 :
  Pat<(VHLO_RecvOpV2:$src $token, $channel_id, $channel_type, $is_host_transfer, VHLO_EmptyTensor:$source_target_pairs),
      (VHLO_RecvOpV1:$dst $token, $channel_id, $channel_type, $is_host_transfer),
      [], [(VHLO_CopyDiscardableAttrs (VHLO_GetFirstOperand $src), (VHLO_GetFirstOperand $dst))]>;

def RecvOpUpgradeV1ToV2 :
  Pat<(VHLO_RecvOpV1:$src $token, $channel_id, $channel_type, $is_host_transfer),
      (VHLO_RecvOpV2:$dst $token, $channel_id, $channel_type, $is_host_transfer, (VHLO_GetEmptyTensor)),
      [], [(VHLO_CopyDiscardableAttrs (VHLO_GetFirstOperand $src), (VHLO_GetFirstOperand $dst))]>;

foreach resultAccuracyOpV1V2Pair = [
  [VHLO_CbrtOpV1, VHLO_CbrtOpV2],
  [VHLO_CosineOpV1, VHLO_CosineOpV2],
  [VHLO_ExpOpV1, VHLO_ExpOpV2],
  [VHLO_Expm1OpV1, VHLO_Expm1OpV2],
  [VHLO_LogOpV1, VHLO_LogOpV2],
  [VHLO_Log1pOpV1, VHLO_Log1pOpV2],
  [VHLO_LogisticOpV1, VHLO_LogisticOpV2],
  [VHLO_RsqrtOpV1, VHLO_RsqrtOpV2],
  [VHLO_SineOpV1, VHLO_SineOpV2],
  [VHLO_SqrtOpV1, VHLO_SqrtOpV2],
  [VHLO_TanOpV1, VHLO_TanOpV2],
  [VHLO_TanhOpV1, VHLO_TanhOpV2]] in {
  def : Pat<(resultAccuracyOpV1V2Pair[0]:$src $operand),
            (resultAccuracyOpV1V2Pair[1]:$dst $operand, (VHLO_GetDefaultResultAccuracyAttr)),
            [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;
  def : Pat<(resultAccuracyOpV1V2Pair[1]:$src $operand, VHLO_DefaultResultAccuracy:$result_accuracy),
            (resultAccuracyOpV1V2Pair[0]:$dst $operand),
            [], [(VHLO_CopyDiscardableAttrs $src, $dst)]>;
}
