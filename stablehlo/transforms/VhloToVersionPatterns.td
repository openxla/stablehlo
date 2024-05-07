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

def ScatterOpUpgradeV1ToV2:
  Pat<(VHLO_ScatterOpV1 $inputs, $scatter_indices, $updates, $update_window_dims, $inserted_window_dims, $scatter_dims_to_operand_dims, $index_vector_dim, $indices_are_sorted, $unique_indices),
      (VHLO_ScatterOpV2 $inputs, $scatter_indices, $updates, $update_window_dims, $inserted_window_dims, (VHLO_GetEmptyDims), (VHLO_GetEmptyDims), $scatter_dims_to_operand_dims, $index_vector_dim, $indices_are_sorted, $unique_indices)>;

def ScatterOpDowngradeV2ToV1 :
  Pat<(VHLO_ScatterOpV2 $inputs, $scatter_indices, $updates, $update_window_dims, $inserted_window_dims, VHLO_EmptyDims:$input_batching_dims, VHLO_EmptyDims:$scatter_indices_batching_dims, $scatter_dims_to_operand_dims, $index_vector_dim, $indices_are_sorted, $unique_indices),
      (VHLO_ScatterOpV1 $inputs, $scatter_indices, $updates, $update_window_dims, $inserted_window_dims, $scatter_dims_to_operand_dims, $index_vector_dim, $indices_are_sorted, $unique_indices)>;
