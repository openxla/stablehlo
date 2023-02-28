/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_VHLO_OPS
#define STABLEHLO_DIALECT_VHLO_OPS

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/VhloDialect.td"
include "stablehlo/dialect/VhloTypes.td"
include "stablehlo/dialect/VhloEnums.td"
include "stablehlo/dialect/VhloAttrs.td"

def VHLO_VersionedOpInterface : OpInterface<"VersionedOpInterface"> {
  let cppNamespace = "::mlir::vhlo";
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMaxVersion">,
  ];
}

class VHLO_Op<string mnemonic, string minVersion, string maxVersion, list<Trait> traits = []> :
    Op<VHLO_Dialect, mnemonic,
      [DeclareOpInterfaceMethods<VHLO_VersionedOpInterface>] # traits> {
  let extraClassDefinition = [{
    mlir::vhlo::Version $cppClass::getMinVersion() {
      auto version = mlir::vhlo::Version::fromString("}] #  minVersion # [{");
      if (failed(version)) llvm::report_fatal_error("invalid version }] # minVersion # [{ in }] # mnemonic # [{");
      return *version;
    }
    mlir::vhlo::Version $cppClass::getMaxVersion() {
      if (!strcmp("}] # maxVersion # [{", "current")) return Version::getCurrentVersion();
      auto version = mlir::vhlo::Version::fromString("}] #  maxVersion # [{");
      if (failed(version)) llvm::report_fatal_error("invalid version }] # maxVersion # [{ in }] # mnemonic # [{");
      return *version;
    }
  }];
}

// Design principles for VHLO ops:
// 1) VHLO is a "shallow" versioned copy of the StableHLO dialect that has been
//    simplified down to a bare minimum. As a result, the main design principle
//    is radical minimalism.
// 2) Don't use traits, except when things will break otherwise, i.e.:
//      * ReturnOp needs a trait for Terminator.
//      * ReduceOp/ReduceWindowOp/ScatterOp need a trait since they have
//      * multiple variadic arguments.
// 3) Use VHLO_AnyType or Variadic<VHLO_AnyType> for all operands and results.
// 4) Use VHLO_AnyAttr for all attributes.
// 5) Use VHLO_AnyRegion for all regions.
// 6) Don't use verifiers, shape functions, etc. The only exception is
//    assembly format for FuncOp, because printouts get unreadable otherwise.

def VHLO_AbsOpV1 : VHLO_Op<"abs_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AddOpV1 : VHLO_Op<"add_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AfterAllOpV1 : VHLO_Op<"after_all_v1", "0.9.0", "current"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$inputs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AllGatherOpV1 : VHLO_Op<"all_gather_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$all_gather_dim,
    VHLO_AnyAttr:$replica_groups,
    VHLO_AnyAttr:$channel_id,
    VHLO_AnyAttr:$use_global_device_ids
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AllReduceOpV1 : VHLO_Op<"all_reduce_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$replica_groups,
    VHLO_AnyAttr:$channel_id,
    VHLO_AnyAttr:$use_global_device_ids
  );
  let regions = (region VHLO_AnyRegion:$computation);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AllToAllOpV1 : VHLO_Op<"all_to_all_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$split_dimension,
    VHLO_AnyAttr:$concat_dimension,
    VHLO_AnyAttr:$split_count,
    VHLO_AnyAttr:$replica_groups,
    VHLO_AnyAttr:$channel_id
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AndOpV1 : VHLO_Op<"and_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_Atan2OpV1 : VHLO_Op<"atan2_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_BatchNormGradOpV1 : VHLO_Op<"batch_norm_grad_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$scale,
    VHLO_AnyType:$mean,
    VHLO_AnyType:$variance,
    VHLO_AnyType:$grad_output,
    VHLO_AnyAttr:$epsilon,
    VHLO_AnyAttr:$feature_index
  );
  let results = (outs
    VHLO_AnyType:$grad_operand,
    VHLO_AnyType:$grad_scale,
    VHLO_AnyType:$grad_offset
  );
}

def VHLO_BatchNormInferenceOpV1 : VHLO_Op<"batch_norm_inference_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$scale,
    VHLO_AnyType:$offset,
    VHLO_AnyType:$mean,
    VHLO_AnyType:$variance,
    VHLO_AnyAttr:$epsilon,
    VHLO_AnyAttr:$feature_index
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_BatchNormTrainingOpV1 : VHLO_Op<"batch_norm_training_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$scale,
    VHLO_AnyType:$offset,
    VHLO_AnyAttr:$epsilon,
    VHLO_AnyAttr:$feature_index
  );
  let results = (outs
    VHLO_AnyType:$output,
    VHLO_AnyType:$batch_mean,
    VHLO_AnyType:$batch_var
  );
}

def VHLO_BitcastConvertOpV1 : VHLO_Op<"bitcast_convert_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_BroadcastInDimOpV1 : VHLO_Op<"broadcast_in_dim_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$broadcast_dimensions
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): BroadcastOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_BroadcastOpV1 : VHLO_Op<"broadcast_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$broadcast_sizes
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#425): CallOp is not part of the StableHLO spec.
// We're planning to look into it as part of the work on speccing
// Module/Func/Call/Return ops in StableHLO.
def VHLO_CallOpV1 : VHLO_Op<"call_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyAttr:$callee, Variadic<VHLO_AnyType>:$operands);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_CaseOpV1 : VHLO_Op<"case_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$index);
  let regions = (region VariadicRegion<VHLO_AnyRegion>:$branches);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_CbrtOpV1 : VHLO_Op<"cbrt_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CeilOpV1 : VHLO_Op<"ceil_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CholeskyOpV1 : VHLO_Op<"cholesky_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$a,
    VHLO_AnyAttr:$lower
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ClampOpV1 : VHLO_Op<"clamp_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$min,
    VHLO_AnyType:$operand,
    VHLO_AnyType:$max
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ClzOpV1 : VHLO_Op<"count_leading_zeros_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CollectivePermuteOpV1 : VHLO_Op<"collective_permute_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$source_target_pairs,
    VHLO_AnyAttr:$channel_id
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CompareOpV1 : VHLO_Op<"compare_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$comparison_direction,
    VHLO_AnyAttr:$compare_type
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ComplexOpV1 : VHLO_Op<"complex_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): ComputeReshapeShapeOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_ComputeReshapeShapeOpV1 : VHLO_Op<"compute_reshape_shape_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$num_elements, VHLO_AnyType:$dynamic_shape);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ConcatenateOpV1 : VHLO_Op<"concatenate_v1", "0.9.0", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ConstantOpV1 : VHLO_Op<"constant_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyAttr:$value);
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_ConvertOpV1 : VHLO_Op<"convert_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ConvolutionOpV1 : VHLO_Op<"convolution_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$window_strides,
    VHLO_AnyAttr:$padding,
    VHLO_AnyAttr:$lhs_dilation,
    VHLO_AnyAttr:$rhs_dilation,
    VHLO_AnyAttr:$window_reversal,
    VHLO_AnyAttr:$input_batch_dimension,
    VHLO_AnyAttr:$input_feature_dimension,
    VHLO_AnyAttr:$input_spatial_dimensions,
    VHLO_AnyAttr:$kernel_input_feature_dimension,
    VHLO_AnyAttr:$kernel_output_feature_dimension,
    VHLO_AnyAttr:$kernel_spatial_dimensions,
    VHLO_AnyAttr:$output_batch_dimension,
    VHLO_AnyAttr:$output_feature_dimension,
    VHLO_AnyAttr:$output_spatial_dimensions,
    VHLO_AnyAttr:$feature_group_count,
    VHLO_AnyAttr:$batch_group_count,
    VHLO_AnyAttr:$precision_config
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CosineOpV1 : VHLO_Op<"cosine_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): CreateTokenOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_CreateTokenOpV1 : VHLO_Op<"create_token_v1", "0.9.0", "current"> {
  let results = (outs VHLO_AnyType:$output);
}

// TODO(#3): CrossReplicaSumOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_CrossReplicaSumOpV1 : VHLO_Op<"cross-replica-sum_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$replica_groups
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): CstrReshapableOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_CstrReshapableOpV1 : VHLO_Op<"cstr_reshapable_v1", "0.9.0", "current"> {
  let results = (outs VHLO_AnyType:$result);
  let arguments = (ins VHLO_AnyType:$num_elements, VHLO_AnyType:$dynamic_shape);
}

// TODO(#1187): api_version is different between VHLO and the StableHLO spec:
// in VHLO it's an enum, in the spec it's an si32.
// TODO(#629): operand_layouts/result_layouts are not part of the spec.
// TODO(#740): output_operand_aliases is not part of the spec.
// CustomCallOp has proven to be one of the trickiest ops to fully spec.
// We're aiming to address all these todos by the release of StableHLO v1.0.
def VHLO_CustomCallOpV1 : VHLO_Op<"custom_call_v1", "0.9.0", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$call_target_name,
    VHLO_AnyAttr:$has_side_effect,
    VHLO_AnyAttr:$backend_config,
    VHLO_AnyAttr:$api_version,
    VHLO_AnyAttr:$called_computations,
    VHLO_AnyAttr:$operand_layouts,
    VHLO_AnyAttr:$result_layouts,
    VHLO_AnyAttr:$output_operand_aliases
  );
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_DivOpV1 : VHLO_Op<"divide_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DotGeneralOpV1 : VHLO_Op<"dot_general_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$lhs_batching_dimensions,
    VHLO_AnyAttr:$rhs_batching_dimensions,
    VHLO_AnyAttr:$lhs_contracting_dimensions,
    VHLO_AnyAttr:$rhs_contracting_dimensions,
    VHLO_AnyAttr:$precision_config
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): DotOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_DotOpV1 : VHLO_Op<"dot_v1", "0.9.0", "current"> {
  let arguments = (
    ins VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$precision_config
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): DynamicBroadcastInDimOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_DynamicBroadcastInDimOpV1 : VHLO_Op<"dynamic_broadcast_in_dim_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$output_dimensions,
    VHLO_AnyAttr:$broadcast_dimensions,
    VHLO_AnyAttr:$known_expanding_dimensions,
    VHLO_AnyAttr:$known_nonexpanding_dimensions
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): DynamicConvOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_DynamicConvOpV1 : VHLO_Op<"dynamic_conv_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyType:$d_padding,
    VHLO_AnyAttr:$window_strides,
    VHLO_AnyAttr:$padding,
    VHLO_AnyAttr:$lhs_dilation,
    VHLO_AnyAttr:$rhs_dilation,
    VHLO_AnyAttr:$window_reversal,
    VHLO_AnyAttr:$input_batch_dimension,
    VHLO_AnyAttr:$input_feature_dimension,
    VHLO_AnyAttr:$input_spatial_dimensions,
    VHLO_AnyAttr:$kernel_input_feature_dimension,
    VHLO_AnyAttr:$kernel_output_feature_dimension,
    VHLO_AnyAttr:$kernel_spatial_dimensions,
    VHLO_AnyAttr:$output_batch_dimension,
    VHLO_AnyAttr:$output_feature_dimension,
    VHLO_AnyAttr:$output_spatial_dimensions,
    VHLO_AnyAttr:$feature_group_count,
    VHLO_AnyAttr:$batch_group_count,
    VHLO_AnyAttr:$precision_config
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): DynamicGatherOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_DynamicGatherOpV1 : VHLO_Op<"dynamic_gather_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$start_indices,
    VHLO_AnyType:$slice_sizes,
    VHLO_AnyAttr:$offset_dims,
    VHLO_AnyAttr:$collapsed_slice_dims,
    VHLO_AnyAttr:$start_index_map,
    VHLO_AnyAttr:$index_vector_dim,
    VHLO_AnyAttr:$indices_are_sorted
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): DynamicIotaOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_DynamicIotaOpV1 : VHLO_Op<"dynamic_iota_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$output_shape, VHLO_AnyAttr:$iota_dimension);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): DynamicPadOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_DynamicPadOpV1 : VHLO_Op<"dynamic_pad_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$padding_value,
    VHLO_AnyType:$edge_padding_low,
    VHLO_AnyType:$edge_padding_high,
    VHLO_AnyType:$interior_padding
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): DynamicReshapeOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_DynamicReshapeOpV1 : VHLO_Op<"dynamic_reshape_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand, VHLO_AnyType:$output_shape);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicSliceOpV1 : VHLO_Op<"dynamic_slice_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    Variadic<VHLO_AnyType>:$start_indices,
    VHLO_AnyAttr:$slice_sizes
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicUpdateSliceOpV1 : VHLO_Op<"dynamic_update_slice_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$update,
    Variadic<VHLO_AnyType>:$start_indices
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): EinsumOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_EinsumOpV1 : VHLO_Op<"einsum_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$einsum_config
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_Expm1OpV1 : VHLO_Op<"exponential_minus_one_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ExpOpV1 : VHLO_Op<"exponential_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_FftOpV1 : VHLO_Op<"fft_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$fft_type,
    VHLO_AnyAttr:$fft_length
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_FloorOpV1 : VHLO_Op<"floor_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#425): FuncOp is not part of the StableHLO spec.
// We're planning to look into it as part of the work on speccing
// Module/Func/Call/Return ops in StableHLO.
def VHLO_FuncOpV1 : VHLO_Op<"func_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyAttr:$sym_name,
    VHLO_AnyAttr:$function_type,
    VHLO_AnyAttr:$sym_visibility,
    VHLO_AnyAttr:$arg_attrs,
    VHLO_AnyAttr:$res_attrs);
  let regions = (region VHLO_AnyRegion:$body);
  let assemblyFormat = "custom<FunctionBody>($sym_name, $body, $function_type) attr-dict";
}

def VHLO_GatherOpV1 : VHLO_Op<"gather_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$start_indices,
    VHLO_AnyAttr:$offset_dims,
    VHLO_AnyAttr:$collapsed_slice_dims,
    VHLO_AnyAttr:$start_index_map,
    VHLO_AnyAttr:$index_vector_dim,
    VHLO_AnyAttr:$slice_sizes,
    VHLO_AnyAttr:$indices_are_sorted
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_GetDimensionSizeOpV1 : VHLO_Op<"get_dimension_size_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_GetTupleElementOpV1 : VHLO_Op<"get_tuple_element_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$index
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_IfOpV1 : VHLO_Op<"if_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$pred);
  let regions = (region
    VHLO_AnyRegion:$true_branch,
    VHLO_AnyRegion:$false_branch
  );
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_ImagOpV1 : VHLO_Op<"imag_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#629): layout is not part of the StableHLO spec.
// The notion of layouts is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_InfeedOpV1 : VHLO_Op<"infeed_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$infeed_config,
    VHLO_AnyAttr:$layout
  );
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_IotaOpV1 : VHLO_Op<"iota_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyAttr:$iota_dimension);
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_IsFiniteOpV1 : VHLO_Op<"is_finite_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$x);
  let results = (outs VHLO_AnyType:$y);
}

def VHLO_Log1pOpV1 : VHLO_Op<"log_plus_one_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_LogisticOpV1 : VHLO_Op<"logistic_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_LogOpV1 : VHLO_Op<"log_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_MapOpV1 : VHLO_Op<"map_v1", "0.9.0", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimensions
  );
  let regions = (region VHLO_AnyRegion:$computation);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_MaxOpV1 : VHLO_Op<"maximum_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_MinOpV1 : VHLO_Op<"minimum_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_MulOpV1 : VHLO_Op<"multiply_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_NegOpV1 : VHLO_Op<"negate_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_NotOpV1 : VHLO_Op<"not_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_OptimizationBarrierOpV1 : VHLO_Op<"optimization_barrier_v1", "0.9.0", "current"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$operand);
  let results = (outs Variadic<VHLO_AnyType>:$result);
}

def VHLO_OrOpV1 : VHLO_Op<"or_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#629): layout is not part of the StableHLO spec.
// The notion of layouts is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_OutfeedOpV1 : VHLO_Op<"outfeed_v1", "0.9.0", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$outfeed_config
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PadOpV1 : VHLO_Op<"pad_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$padding_value,
    VHLO_AnyAttr:$edge_padding_low,
    VHLO_AnyAttr:$edge_padding_high,
    VHLO_AnyAttr:$interior_padding
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PartitionIdOpV1 : VHLO_Op<"partition_id_v1", "0.9.0", "current"> {
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PopulationCountOpV1 : VHLO_Op<"popcnt_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PowOpV1 : VHLO_Op<"power_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): RealDynamicSliceOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_RealDynamicSliceOpV1 : VHLO_Op<"real_dynamic_slice_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$start_indices,
    VHLO_AnyType:$limit_indices,
    VHLO_AnyType:$strides
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RealOpV1 : VHLO_Op<"real_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RecvOpV1 : VHLO_Op<"recv_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$channel_id,
    VHLO_AnyAttr:$channel_type,
    VHLO_AnyAttr:$is_host_transfer
  );
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_ReduceOpV1 : VHLO_Op<"reduce_v1", "0.9.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    Variadic<VHLO_AnyType>:$init_values,
    VHLO_AnyAttr:$dimensions
  );
  let regions = (region VHLO_AnyRegion:$body);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_ReducePrecisionOpV1 : VHLO_Op<"reduce_precision_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$exponent_bits,
    VHLO_AnyAttr:$mantissa_bits
  );
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_ReduceScatterOpV1 : VHLO_Op<"reduce_scatter_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$scatter_dimension,
    VHLO_AnyAttr:$replica_groups,
    VHLO_AnyAttr:$channel_id,
    VHLO_AnyAttr:$use_global_device_ids
  );
  let regions = (region VHLO_AnyRegion:$computation);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ReduceWindowOpV1 : VHLO_Op<"reduce_window_v1", "0.9.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    Variadic<VHLO_AnyType>:$init_values,
    VHLO_AnyAttr:$window_dimensions,
    VHLO_AnyAttr:$window_strides,
    VHLO_AnyAttr:$base_dilations,
    VHLO_AnyAttr:$window_dilations,
    VHLO_AnyAttr:$padding
  );
  let regions = (region VHLO_AnyRegion:$body);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_RemOpV1 : VHLO_Op<"remainder_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ReplicaIdOpV1 : VHLO_Op<"replica_id_v1", "0.9.0", "current"> {
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ReshapeOpV1 : VHLO_Op<"reshape_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#425): ReturnOp is not part of the StableHLO spec.
// We're planning to look into it as part of the work on speccing
// Module/Func/Call/Return ops in StableHLO.
def VHLO_ReturnOpV1 : VHLO_Op<"return_v1", "0.9.0", "current", [Terminator]> {
  let arguments = (ins Variadic<VHLO_AnyType>:$results);
}

def VHLO_ReverseOpV1 : VHLO_Op<"reverse_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$dimensions
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RngBitGeneratorOpV1 : VHLO_Op<"rng_bit_generator_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyAttr:$rng_algorithm,
    VHLO_AnyType:$initial_state
  );
  let results = (outs
    VHLO_AnyType:$output_state,
    VHLO_AnyType:$output
  );
}

def VHLO_RngOpV1 : VHLO_Op<"rng_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$a,
    VHLO_AnyType:$b,
    VHLO_AnyType:$shape,
    VHLO_AnyAttr:$rng_distribution
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RoundNearestEvenOpV1 : VHLO_Op<"round_nearest_even_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RoundOpV1 : VHLO_Op<"round_nearest_afz_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RsqrtOpV1 : VHLO_Op<"rsqrt_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ScatterOpV1 : VHLO_Op<"scatter_v1", "0.9.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyType:$scatter_indices,
    Variadic<VHLO_AnyType>:$updates,
    VHLO_AnyAttr:$update_window_dims,
    VHLO_AnyAttr:$inserted_window_dims,
    VHLO_AnyAttr:$scatter_dims_to_operand_dims,
    VHLO_AnyAttr:$index_vector_dim,
    VHLO_AnyAttr:$indices_are_sorted,
    VHLO_AnyAttr:$unique_indices
  );
  let regions = (region VHLO_AnyRegion:$update_computation);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_SelectAndScatterOpV1 : VHLO_Op<"select_and_scatter_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$source,
    VHLO_AnyType:$init_value,
    VHLO_AnyAttr:$window_dimensions,
    VHLO_AnyAttr:$window_strides,
    VHLO_AnyAttr:$padding
  );
  let regions = (region VHLO_AnyRegion:$select, VHLO_AnyRegion:$scatter);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SelectOpV1 : VHLO_Op<"select_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$pred,
    VHLO_AnyType:$on_true,
    VHLO_AnyType:$on_false
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SendOpV1 : VHLO_Op<"send_v1", "0.9.0", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$channel_id,
    VHLO_AnyAttr:$channel_type,
    VHLO_AnyAttr:$is_host_transfer
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SetDimensionSizeOpV1 : VHLO_Op<"set_dimension_size_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$size,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ShiftLeftOpV1 : VHLO_Op<"shift_left_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ShiftRightArithmeticOpV1 : VHLO_Op<"shift_right_arithmetic_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ShiftRightLogicalOpV1 : VHLO_Op<"shift_right_logical_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SignOpV1 : VHLO_Op<"sign_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SineOpV1 : VHLO_Op<"sine_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SliceOpV1 : VHLO_Op<"slice_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$start_indices,
    VHLO_AnyAttr:$limit_indices,
    VHLO_AnyAttr:$strides
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SortOpV1 : VHLO_Op<"sort_v1", "0.9.0", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimension,
    VHLO_AnyAttr:$is_stable
  );
  let regions = (region VHLO_AnyRegion:$comparator);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_SqrtOpV1 : VHLO_Op<"sqrt_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SubtractOpV1 : VHLO_Op<"subtract_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_TanhOpV1 : VHLO_Op<"tanh_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): TorchIndexSelectOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_TorchIndexSelectOpV1 : VHLO_Op<"torch_index_select_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$index,
    VHLO_AnyAttr:$dim,
    VHLO_AnyAttr:$batch_dims
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): TraceOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_TraceOpV1 : VHLO_Op<"trace_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$tag
  );
}

def VHLO_TransposeOpV1 : VHLO_Op<"transpose_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$permutation
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_TriangularSolveOpV1 : VHLO_Op<"triangular_solve_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$a,
    VHLO_AnyType:$b,
    VHLO_AnyAttr:$left_side,
    VHLO_AnyAttr:$lower,
    VHLO_AnyAttr:$unit_diagonal,
    VHLO_AnyAttr:$transpose_a
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_TupleOpV1 : VHLO_Op<"tuple_v1", "0.9.0", "current"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$val);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): UnaryEinsumOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_UnaryEinsumOpV1 : VHLO_Op<"unary_einsum_v1", "0.9.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$einsum_config
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#588): UniformDequantizeOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_UniformDequantizeOpV1 : VHLO_Op<"uniform_dequantize_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#588): UniformQuantizeOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_UniformQuantizeOpV1 : VHLO_Op<"uniform_quantize_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_WhileOpV1 : VHLO_Op<"while_v1", "0.9.0", "current"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$operand);
  let regions = (region VHLO_AnyRegion:$cond, VHLO_AnyRegion:$body);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_XorOpV1 : VHLO_Op<"xor_v1", "0.9.0", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

#endif // STABLEHLO_DIALECT_STABLEHLO_OPS
