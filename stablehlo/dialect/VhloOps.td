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

//===----------------------------------------------------------------------===//
// Dialect and Ops
//===----------------------------------------------------------------------===//

include "stablehlo/dialect/VhloDialect.td"
include "stablehlo/dialect/VhloTypes.td"
include "stablehlo/dialect/VhloEnums.td"
include "stablehlo/dialect/VhloAttrs.td"

def VersionedOpInterface : OpInterface<"VersionedOpInterface"> {
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMaxVersion">,
  ];
}

// Most ops should not use traits. Exceptions are:
// - ReturnOp needs a trait for Terminator.
// - ReduceOp/ReduceWindowOp/ScatterOp need a trait since they have
//   multiple variadic arguments.
class VHLO_Op<string mnemonic, string minVersion = "0.3.0", string maxVersion = "current", list<Trait> traits = []> :
    Op<VHLO_Dialect, mnemonic,
      [DeclareOpInterfaceMethods<VersionedOpInterface>] # traits> {
  let extraClassDefinition = [{
    mlir::vhlo::Version $cppClass::getMinVersion() {
      auto version = mlir::vhlo::Version::fromString("}] #  minVersion # [{");
      if (failed(version)) llvm_unreachable("invalid version }] # minVersion # [{ in }] # mnemonic # [{");
      return *version;
    }
    mlir::vhlo::Version $cppClass::getMaxVersion() {
      if (!strcmp("}] # maxVersion # [{", "current")) return Version::getCurrentVersion();
      auto version = mlir::vhlo::Version::fromString("}] #  maxVersion # [{");
      if (failed(version)) llvm_unreachable("invalid version }] # maxVersion # [{ in }] # mnemonic # [{");
      return *version;
    }
  }];
}

def VHLO_ConstantOpV1 : VHLO_Op<"constant"> {
  let arguments = (ins VHLO_AnyAttr:$value);
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_IotaOpV1 : VHLO_Op<"iota"> {
  let arguments = (ins VHLO_AnyAttr:$iota_dimension);
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_DynamicIotaOpV1 : VHLO_Op<"dynamic_iota"> {
  let arguments = (ins VHLO_AnyType:$output_shape, VHLO_AnyAttr:$iota_dimension);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CreateTokenOpV1 : VHLO_Op<"create_token"> {
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_AbsOpV1 : VHLO_Op<"abs"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CbrtOpV1 : VHLO_Op<"cbrt"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CeilOpV1 : VHLO_Op<"ceil"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ConvertOpV1 : VHLO_Op<"convert"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ClzOpV1 : VHLO_Op<"count_leading_zeros"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CosineOpV1 : VHLO_Op<"cosine"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ExpOpV1 : VHLO_Op<"exponential"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_Expm1OpV1 : VHLO_Op<"exponential_minus_one"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_FloorOpV1 : VHLO_Op<"floor"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ImagOpV1 : VHLO_Op<"imag"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_IsFiniteOpV1 : VHLO_Op<"is_finite"> {
  let arguments = (ins VHLO_AnyType:$x);
  let results = (outs VHLO_AnyType:$y);
}

def VHLO_LogOpV1 : VHLO_Op<"log">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_Log1pOpV1 : VHLO_Op<"log_plus_one">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_LogisticOpV1 : VHLO_Op<"logistic">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_NotOpV1 : VHLO_Op<"not">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_NegOpV1 : VHLO_Op<"negate">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PopulationCountOpV1 : VHLO_Op<"popcnt">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RealOpV1 : VHLO_Op<"real">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RoundOpV1 : VHLO_Op<"round_nearest_afz">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RoundNearestEvenOpV1 : VHLO_Op<"round_nearest_even">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RsqrtOpV1 : VHLO_Op<"rsqrt">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SignOpV1 : VHLO_Op<"sign">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SineOpV1 : VHLO_Op<"sine">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SqrtOpV1 : VHLO_Op<"sqrt">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_TanhOpV1 : VHLO_Op<"tanh">{
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// Binary Ops
def VHLO_AddOpV1 : VHLO_Op<"add"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_Atan2OpV1 : VHLO_Op<"atan2"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_ComplexOpV1 : VHLO_Op<"complex"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_DivOpV1 : VHLO_Op<"divide"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_MaxOpV1 : VHLO_Op<"maximum"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_MinOpV1 : VHLO_Op<"minimum"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_MulOpV1 : VHLO_Op<"multiply"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_PowOpV1 : VHLO_Op<"power"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_RemOpV1 : VHLO_Op<"remainder"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_ShiftLeftOpV1 : VHLO_Op<"shift_left"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_ShiftRightArithmeticOpV1 : VHLO_Op<"shift_right_arithmetic"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_ShiftRightLogicalOpV1 : VHLO_Op<"shift_right_logical"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_SubtractOpV1 : VHLO_Op<"subtract"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

// Logical Ops
def VHLO_AndOpV1 : VHLO_Op<"and"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_OrOpV1 : VHLO_Op<"or"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}
def VHLO_XorOpV1 : VHLO_Op<"xor"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ReturnOpV1 : VHLO_Op<"return", "0.3.0", "current", [Terminator]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$results
  );
  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

// Communication op definitions.
def VHLO_InfeedOpV1 : VHLO_Op<"infeed"> {
  let arguments = (ins
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$infeed_config,
    OptionalAttr<VHLO_AnyAttr>:$layout
  );
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_OutfeedOpV1 : VHLO_Op<"outfeed"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$outfeed_config
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_SendOpV1 : VHLO_Op<"send"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$channel_handle,
    OptionalAttr<VHLO_AnyAttr>:$is_host_transfer
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_RecvOpV1 : VHLO_Op<"recv"> {
  let arguments = (ins
    VHLO_AnyType:$token,
    VHLO_AnyAttr:$channel_handle,
    OptionalAttr<VHLO_AnyAttr>:$is_host_transfer
  );
  let results = (outs Variadic<VHLO_AnyType>);
}

// Parallelism related op definitions.
def VHLO_ReplicaIdOpV1 : VHLO_Op<"replica_id"> {
  let results = (outs VHLO_AnyType);
}

def VHLO_PartitionIdOpV1 : VHLO_Op<"partition_id", "0.4.0"> {
  let results = (outs VHLO_AnyType);
}

// Control flow op definitions.
def VHLO_AfterAllOpV1 : VHLO_Op<"after_all"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$inputs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_IfOpV1 : VHLO_Op<"if"> {
  let arguments = (ins VHLO_AnyType:$pred);
  let regions = (region VHLO_AnyRegion:$true_branch,
                        VHLO_AnyRegion:$false_branch);
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_CaseOpV1 : VHLO_Op<"case"> {
  let arguments = (ins VHLO_AnyType:$index);
  let regions = (region VariadicRegion<VHLO_AnyRegion>:$branches);
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_WhileOpV1 : VHLO_Op<"while"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$operand);
  let regions = (region VHLO_AnyRegion:$cond, VHLO_AnyRegion:$body);
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_AllGatherOpV1 : VHLO_Op<"all_gather", "0.3.0", "0.3.0"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$all_gather_dim,
    VHLO_AnyAttr:$replica_groups,
    OptionalAttr<VHLO_AnyAttr>:$channel_handle
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_AllGatherOpV2 : VHLO_Op<"all_gather_v2", "0.4.0"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$all_gather_dim,
    VHLO_AnyAttr:$replica_groups,
    OptionalAttr<VHLO_AnyAttr>:$channel_handle,
    OptionalAttr<VHLO_AnyAttr>:$use_global_device_ids
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_AllReduceOpV1 : VHLO_Op<"all_reduce"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$replica_groups,
    OptionalAttr<VHLO_AnyAttr>:$channel_handle,
    OptionalAttr<VHLO_AnyAttr>:$use_global_device_ids
  );
  let regions = (region VHLO_AnyRegion:$computation);
  let results = (outs VHLO_AnyType);
}

def VHLO_ReduceScatterOpV1 : VHLO_Op<"reduce_scatter"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$scatter_dimension,
    VHLO_AnyAttr:$replica_groups,
    OptionalAttr<VHLO_AnyAttr>:$channel_handle,
    OptionalAttr<VHLO_AnyAttr>:$use_global_device_ids
  );
  let regions = (region VHLO_AnyRegion:$computation);
  let results = (outs VHLO_AnyType);
}

def VHLO_AllToAllOpV1 : VHLO_Op<"all_to_all", "0.3.0", "0.3.0"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$split_dimension,
    VHLO_AnyAttr:$concat_dimension,
    VHLO_AnyAttr:$split_count,
    VHLO_AnyAttr:$replica_groups
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_AllToAllOpV2 : VHLO_Op<"all_to_all_v2", "0.4.0"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$split_dimension,
    VHLO_AnyAttr:$concat_dimension,
    VHLO_AnyAttr:$split_count,
    VHLO_AnyAttr:$replica_groups,
    OptionalAttr<VHLO_AnyAttr>:$channel_handle
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_ReduceOpV1 : VHLO_Op<"reduce", "0.3.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    Variadic<VHLO_AnyType>:$init_values,
    VHLO_AnyAttr:$dimensions
  );
  let results = (outs Variadic<VHLO_AnyType>);
  let regions = (region VHLO_AnyRegion:$body);
}

//===----------------------------------------------------------------------===//
// VHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def VHLO_GetTupleElementOpV1 : VHLO_Op<"get_tuple_element"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$index
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_TupleOpV1 : VHLO_Op<"tuple"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$val);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CompareOpV1 : VHLO_Op<"compare"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$comparison_direction,
    OptionalAttr<VHLO_AnyAttr>:$compare_type
  );
  let results = (outs VHLO_AnyType);
}

// Slice ops
def VHLO_SliceOpV1 : VHLO_Op<"slice"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$start_indices,
    VHLO_AnyAttr:$limit_indices,
    VHLO_AnyAttr:$strides
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_DynamicSliceOpV1 : VHLO_Op<"dynamic_slice"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    Variadic<VHLO_AnyType>:$start_indices,
    VHLO_AnyAttr:$slice_sizes
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicUpdateSliceOpV1 : VHLO_Op<"dynamic_update_slice"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$update,
    Variadic<VHLO_AnyType>:$start_indices
  );
  let results = (outs VHLO_AnyType:$result);
}

// Other op definitions.
def VHLO_BatchNormGradOpV1 : VHLO_Op<"batch_norm_grad"> {
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
      VHLO_AnyType:$grad_offset);
}

def VHLO_BatchNormInferenceOpV1 : VHLO_Op<"batch_norm_inference"> {
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

def VHLO_BatchNormTrainingOpV1 : VHLO_Op<"batch_norm_training"> {
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
      VHLO_AnyType:$batch_var);
}

def VHLO_BitcastConvertOpV1 : VHLO_Op<"bitcast_convert"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType);
}

def VHLO_BroadcastOpV1 : VHLO_Op<"broadcast"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$broadcast_sizes
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_BroadcastInDimOpV1 : VHLO_Op<"broadcast_in_dim"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$broadcast_dimensions
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_DynamicBroadcastInDimOpV1 : VHLO_Op<"dynamic_broadcast_in_dim"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$output_dimensions,
    VHLO_AnyAttr:$broadcast_dimensions,
    OptionalAttr<VHLO_AnyAttr>:$known_expanding_dimensions,
    OptionalAttr<VHLO_AnyAttr>:$known_nonexpanding_dimensions
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_CholeskyOpV1 : VHLO_Op<"cholesky"> {
  let arguments = (ins
    VHLO_AnyType:$a,
    OptionalAttr<VHLO_AnyAttr>:$lower
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ClampOpV1 : VHLO_Op<"clamp"> {
  let arguments = (ins
    VHLO_AnyType:$min,
    VHLO_AnyType:$operand,
    VHLO_AnyType:$max
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ConcatenateOpV1 : VHLO_Op<"concatenate"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_CollectivePermuteOpV1 : VHLO_Op<"collective_permute", "0.3.0", "0.3.0"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$source_target_pairs
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_CollectivePermuteOpV2 : VHLO_Op<"collective_permute_v2", "0.4.0"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$source_target_pairs,
    OptionalAttr<VHLO_AnyAttr>:$channel_handle
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_ConvolutionOpV1 : VHLO_Op<"convolution"> {
  let arguments = !con(
    (ins
       VHLO_AnyType:$lhs,
       VHLO_AnyType:$rhs),
    VHLO_ConvolutionAttributesV1.attributes);
  let results = (outs VHLO_AnyType);
}

def VHLO_CrossReplicaSumOpV1 : VHLO_Op<"cross-replica-sum"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$replica_groups
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_CustomCallOpV1 : VHLO_Op<"custom_call", "0.3.0", "0.3.0"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$call_target_name,
    OptionalAttr<VHLO_AnyAttr>:$has_side_effect,
    VHLO_AnyAttr:$backend_config,
    OptionalAttr<VHLO_AnyAttr>:$api_version,
    OptionalAttr<VHLO_AnyAttr>:$called_computations,
    OptionalAttr<VHLO_AnyAttr>:$operand_layouts,
    OptionalAttr<VHLO_AnyAttr>:$result_layouts
  );
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_CustomCallOpV2: VHLO_Op<"custom_call_v2", "0.4.0"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$call_target_name,
    OptionalAttr<VHLO_AnyAttr>:$has_side_effect,
    VHLO_AnyAttr:$backend_config,
    OptionalAttr<VHLO_AnyAttr>:$api_version,
    OptionalAttr<VHLO_AnyAttr>:$called_computations,
    OptionalAttr<VHLO_AnyAttr>:$operand_layouts,
    OptionalAttr<VHLO_AnyAttr>:$result_layouts,
    OptionalAttr<VHLO_AnyAttr>:$output_operand_aliases
  );
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_DotOpV1 : VHLO_Op<"dot"> {
  let arguments = (
    ins VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    OptionalAttr<VHLO_AnyAttr>:$precision_config
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_DotGeneralOpV1 : VHLO_Op<"dot_general"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$dot_dimension_numbers,
    OptionalAttr<VHLO_AnyAttr>:$precision_config
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_EinsumOpV1 : VHLO_Op<"einsum"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$einsum_config
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_UnaryEinsumOpV1 : VHLO_Op<"unary_einsum"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$einsum_config
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_FftOpV1 : VHLO_Op<"fft"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$fft_type,
    VHLO_AnyAttr:$fft_length
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_GatherOpV1 : VHLO_Op<"gather"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$start_indices,
    VHLO_AnyAttr:$dimension_numbers,
    VHLO_AnyAttr:$slice_sizes,
    OptionalAttr<VHLO_AnyAttr>:$indices_are_sorted
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_GetDimensionSizeOpV1 : VHLO_Op<"get_dimension_size"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_MapOpV1 : VHLO_Op<"map"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimensions
  );
  let regions = (region VHLO_AnyRegion:$computation);
  let results = (outs VHLO_AnyType);
}

def VHLO_ReshapeOpV1 : VHLO_Op<"reshape"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType);
}

def VHLO_DynamicReshapeOpV1 : VHLO_Op<"dynamic_reshape"> {
  let arguments = (ins VHLO_AnyType:$operand, VHLO_AnyType:$output_shape);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ScatterOpV1 : VHLO_Op<"scatter", "0.3.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyType:$scatter_indices,
    Variadic<VHLO_AnyType>:$updates,
    VHLO_AnyAttr:$scatter_dimension_numbers,
    OptionalAttr<VHLO_AnyAttr>:$indices_are_sorted,
    OptionalAttr<VHLO_AnyAttr>:$unique_indices
  );
  let regions = (region VHLO_AnyRegion:$update_computation);
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_SelectOpV1 : VHLO_Op<"select"> {
  let arguments = (ins
    VHLO_AnyType:$pred,
    VHLO_AnyType:$on_true,
    VHLO_AnyType:$on_false
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SelectAndScatterOpV1 : VHLO_Op<"select_and_scatter"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$source,
    VHLO_AnyType:$init_value,
    OptionalAttr<VHLO_AnyAttr>:$window_dimensions,
    OptionalAttr<VHLO_AnyAttr>:$window_strides,
    OptionalAttr<VHLO_AnyAttr>:$padding
  );
  let regions = (region VHLO_AnyRegion:$select, VHLO_AnyRegion:$scatter);
  let results = (outs VHLO_AnyType);
}

def VHLO_SetDimensionSizeOpV1 : VHLO_Op<"set_dimension_size"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$size,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_SortOpV1 : VHLO_Op<"sort"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    OptionalAttr<VHLO_AnyAttr>:$dimension,
    OptionalAttr<VHLO_AnyAttr>:$is_stable
  );
  let regions = (region VHLO_AnyRegion:$comparator);
  let results = (outs Variadic<VHLO_AnyType>);
}

def VHLO_ReverseOpV1 : VHLO_Op<"reverse"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$dimensions
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PadOpV1 : VHLO_Op<"pad"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$padding_value,
    VHLO_AnyAttr:$edge_padding_low,
    VHLO_AnyAttr:$edge_padding_high,
    VHLO_AnyAttr:$interior_padding
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_TraceOpV1 : VHLO_Op<"trace"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$tag
  );
}

def VHLO_TransposeOpV1 : VHLO_Op<"transpose"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$permutation
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_TriangularSolveOpV1 : VHLO_Op<"triangular_solve"> {
  let arguments = (ins
    VHLO_AnyType:$a,
    VHLO_AnyType:$b,
    VHLO_AnyAttr:$left_side,
    VHLO_AnyAttr:$lower,
    VHLO_AnyAttr:$unit_diagonal,
    VHLO_AnyAttr:$transpose_a
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_ReduceWindowOpV1 : VHLO_Op<"reduce_window", "0.3.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    Variadic<VHLO_AnyType>:$init_values,
    VHLO_AnyAttr:$window_dimensions,
    OptionalAttr<VHLO_AnyAttr>:$window_strides,
    OptionalAttr<VHLO_AnyAttr>:$base_dilations,
    OptionalAttr<VHLO_AnyAttr>:$window_dilations,
    OptionalAttr<VHLO_AnyAttr>:$padding
  );
  let results = (outs Variadic<VHLO_AnyType>);
  let regions = (region VHLO_AnyRegion:$body);
}

def VHLO_TorchIndexSelectOpV1 : VHLO_Op<"torch_index_select"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$index,
    VHLO_AnyAttr:$dim,
    VHLO_AnyAttr:$batch_dims
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_OptimizationBarrierOpV1 : VHLO_Op<"optimization_barrier"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$operand);
  let results = (outs Variadic<VHLO_AnyType>:$result);
}

//===----------------------------------------------------------------------===//
// VHLO RNG Operators.
//===----------------------------------------------------------------------===//

def VHLO_RngOpV1 : VHLO_Op<"rng"> {
  let arguments = (ins
    VHLO_AnyType:$a,
    VHLO_AnyType:$b,
    VHLO_AnyType:$shape,
    VHLO_AnyAttr:$rng_distribution
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_RngBitGeneratorOpV1 : VHLO_Op<"rng_bit_generator"> {
  let arguments = (ins
    VHLO_AnyAttr:$rng_algorithm,
    VHLO_AnyType:$initial_state
  );
  let results = (outs
    VHLO_AnyType:$output_state,
    VHLO_AnyType:$output
  );
}

// Quantize Ops
def VHLO_UniformQuantizeOpV1 : VHLO_Op<"uniform_quantize"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_UniformDequantizeOpV1 : VHLO_Op<"uniform_dequantize"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ReducePrecisionOpV1 : VHLO_Op<"reduce_precision"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$exponent_bits,
    VHLO_AnyAttr:$mantissa_bits
  );
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_RealDynamicSliceOpV1 : VHLO_Op<"real_dynamic_slice"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$start_indices,
    VHLO_AnyType:$limit_indices,
    VHLO_AnyType:$strides
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicPadOpV1 : VHLO_Op<"dynamic_pad"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$padding_value,
    VHLO_AnyType:$edge_padding_low,
    VHLO_AnyType:$edge_padding_high,
    VHLO_AnyType:$interior_padding
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicGatherOpV1 : VHLO_Op<"dynamic_gather"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$start_indices,
    VHLO_AnyType:$slice_sizes,
    VHLO_AnyAttr:$dimension_numbers,
    OptionalAttr<VHLO_AnyAttr>:$indices_are_sorted
  );
  let results = (outs VHLO_AnyType);
}

def VHLO_DynamicConvOpV1 : VHLO_Op<"dynamic_conv"> {
  let arguments = !con(
    (ins
       VHLO_AnyType:$lhs,
       VHLO_AnyType:$rhs,
       VHLO_AnyType:$d_padding),
    VHLO_ConvolutionAttributesV1.attributes);
  let results = (outs VHLO_AnyType);
}

def VHLO_ComputeReshapeShapeOpV1 : VHLO_Op<"compute_reshape_shape"> {
  let arguments = (ins VHLO_AnyType:$num_elements, VHLO_AnyType:$dynamic_shape);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CstrReshapableOpV1 : VHLO_Op<"cstr_reshapable"> {
  let results = (outs VHLO_AnyType:$result);
  let arguments = (ins VHLO_AnyType:$num_elements, VHLO_AnyType:$dynamic_shape);
}

def VHLO_FuncOpV1 : VHLO_Op<"func"> {
  let arguments = (ins VHLO_AnyAttr:$sym_name,
                       VHLO_AnyAttr:$function_type,
                       OptionalAttr<VHLO_AnyAttr>:$sym_visibility,
                       OptionalAttr<VHLO_AnyAttr>:$arg_attrs,
                       OptionalAttr<VHLO_AnyAttr>:$res_attrs);
  let regions = (region VHLO_AnyRegion:$body);
  let assemblyFormat = "custom<FunctionBody>($sym_name, $body, $function_type) attr-dict";
}

def VHLO_CallOpV1 : VHLO_Op<"call"> {
  let arguments = (ins VHLO_AnyAttr:$callee, Variadic<VHLO_AnyType>:$operands);
  let results = (outs Variadic<VHLO_AnyType>);
}

#endif // STABLEHLO_DIALECT_STABLEHLO_OPS
