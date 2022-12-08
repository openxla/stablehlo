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

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect and Ops
//===----------------------------------------------------------------------===//

def VHLO_Dialect : Dialect {
  let name = "vhlo";
  let cppNamespace = "::mlir::vhlo";

  let description = [{
    A shim opset of Versioned StableHLO ops for versions 0.x.x and 1.x.x.

    Version log:
      0.3.0: Bootstrap from MHLO: https://github.com/openxla/stablehlo/pull/1.
      0.4.0: Add AllGatherOp::use_global_device_ids: https://github.com/openxla/stablehlo/pull/272.
             Add CollectivePermuteOp::channel_handle: https://github.com/openxla/stablehlo/pull/388.
             Add CustomCallOp::output_operand_aliases: https://github.com/openxla/stablehlo/pull/403.
             Add PartitionIdOp: https://github.com/openxla/stablehlo/pull/673.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

include "stablehlo/dialect/VhloEnums.td"
include "stablehlo/dialect/VhloAttrs.td"

// Most ops should not use traits. Exceptions are:
// - ReturnOp needs a trait for Terminator.
// - ReduceOp/ReduceWindowOp/ScatterOp need a trait since they have
//   multiple variadic arguments.
class VHLO_Op<string mnemonic, string minVersion = "0.3.0", string maxVersion = "current", list<Trait> traits = []> :
    Op<VHLO_Dialect, mnemonic,
      [DeclareOpInterfaceMethods<VersionedInterface>] # traits> {
  let extraClassDefinition = [{
    mlir::vhlo::Version $cppClass::getMinVersion() {
      return *mlir::vhlo::Version::fromString("}] #  minVersion # [{");
    }
    mlir::vhlo::Version $cppClass::getMaxVersion() {
      if (!strcmp("}] # maxVersion # [{", "current")) return VhloDialect::getCurrentVersion();
      return *mlir::vhlo::Version::fromString("}] #  maxVersion # [{");
    }
  }];
}

def VHLO_ConstantOpV1 : VHLO_Op<"constant"> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs VHLO_StaticShapeTensor:$output);
}

def VHLO_IotaOpV1 : VHLO_Op<"iota"> {
  let arguments = (ins I64Attr:$iota_dimension);
  let results = (outs VHLO_IntFpOrComplexTensor:$output);
}

def VHLO_DynamicIotaOpV1 : VHLO_Op<"dynamic_iota"> {
  let arguments = (ins VHLO_DimensionTensor:$output_shape, I64Attr:$iota_dimension);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_CreateTokenOpV1 : VHLO_Op<"create_token"> {
  let results = (outs VHLO_Token:$output);
}

def VHLO_AbsOpV1 : VHLO_Op<"abs"> {
  let arguments = (ins TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$operand);
  let results = (outs TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$result);
}

def VHLO_CbrtOpV1 : VHLO_Op<"cbrt"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_CeilOpV1 : VHLO_Op<"ceil"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_ConvertOpV1 : VHLO_Op<"convert"> {
  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ClzOpV1 : VHLO_Op<"count_leading_zeros"> {
  let arguments = (ins VHLO_IntTensor:$operand);
  let results = (outs VHLO_IntTensor:$result);
}

def VHLO_CosineOpV1 : VHLO_Op<"cosine"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_ExpOpV1 : VHLO_Op<"exponential"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_Expm1OpV1 : VHLO_Op<"exponential_minus_one"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}
def VHLO_FloorOpV1 : VHLO_Op<"floor"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_ImagOpV1 : VHLO_Op<"imag"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_IsFiniteOpV1 : VHLO_Op<"is_finite"> {
  let arguments = (ins VHLO_FpTensor:$x);
  let results = (outs VHLO_PredTensor:$y);
}

def VHLO_LogOpV1 : VHLO_Op<"log">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_Log1pOpV1 : VHLO_Op<"log_plus_one">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_LogisticOpV1 : VHLO_Op<"logistic">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_NotOpV1 : VHLO_Op<"not">{
  let arguments = (ins VHLO_PredOrIntTensor:$operand);
  let results = (outs VHLO_PredOrIntTensor:$result);
}

def VHLO_NegOpV1 : VHLO_Op<"negate">{
  let arguments = (ins VHLO_IntFpOrComplexTensor:$operand);
  let results = (outs VHLO_IntFpOrComplexTensor:$result);
}

def VHLO_PopulationCountOpV1 : VHLO_Op<"popcnt">{
  let arguments = (ins VHLO_IntTensor:$operand);
  let results = (outs VHLO_IntTensor:$result);
}

def VHLO_RealOpV1 : VHLO_Op<"real">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_RoundOpV1 : VHLO_Op<"round_nearest_afz">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_RoundNearestEvenOpV1 : VHLO_Op<"round_nearest_even">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_RsqrtOpV1 : VHLO_Op<"rsqrt">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_SignOpV1 : VHLO_Op<"sign">{
  let arguments = (ins TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$operand);
  let results = (outs TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$result);
}

def VHLO_SineOpV1 : VHLO_Op<"sine">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_SqrtOpV1 : VHLO_Op<"sqrt">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_TanhOpV1 : VHLO_Op<"tanh">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

// Binary Ops
def VHLO_AddOpV1 : VHLO_Op<"add"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_Atan2OpV1 : VHLO_Op<"atan2"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ComplexOpV1 : VHLO_Op<"complex"> {
  let arguments = (ins VHLO_Fp32Or64Tensor:$lhs, VHLO_Fp32Or64Tensor:$rhs);
  let results = (outs VHLO_ComplexTensor:$result);
}
def VHLO_DivOpV1 : VHLO_Op<"divide"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_MaxOpV1 : VHLO_Op<"maximum"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_MinOpV1 : VHLO_Op<"minimum"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_MulOpV1 : VHLO_Op<"multiply"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_PowOpV1 : VHLO_Op<"power"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_RemOpV1 : VHLO_Op<"remainder"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ShiftLeftOpV1 : VHLO_Op<"shift_left"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ShiftRightArithmeticOpV1 : VHLO_Op<"shift_right_arithmetic"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ShiftRightLogicalOpV1 : VHLO_Op<"shift_right_logical"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_SubtractOpV1 : VHLO_Op<"subtract"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}

// Logical Ops
def VHLO_AndOpV1 : VHLO_Op<"and"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_OrOpV1 : VHLO_Op<"or"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_XorOpV1 : VHLO_Op<"xor"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ReturnOpV1 : VHLO_Op<"return", "0.3.0", "current", [Terminator]> {
  let arguments = (ins
    Variadic<VHLO_TensorOrTokenOrTuple >:$results
  );
  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

// Communication op definitions.
def VHLO_InfeedOpV1 : VHLO_Op<"infeed"> {
  let arguments = (ins
    VHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$infeed_config,
    OptionalAttr<ArrayAttr>:$layout
  );
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_OutfeedOpV1 : VHLO_Op<"outfeed"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    VHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$outfeed_config
  );
  let results = (outs VHLO_Token);
}

def VHLO_SendOpV1 : VHLO_Op<"send"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    VHLO_Token:$token,
    VHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );
  let results = (outs VHLO_Token);
}

def VHLO_RecvOpV1 : VHLO_Op<"recv"> {
  let arguments = (ins
    VHLO_Token:$token,
    VHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

// Parallelism related op definitions.
def VHLO_ReplicaIdOpV1 : VHLO_Op<"replica_id"> {
  let results = (outs TensorOf<[UI32]>);
}

def VHLO_PartitionIdOpV1 : VHLO_Op<"partition_id", "0.4.0"> {
  let results = (outs TensorOf<[UI32]>);
}

// Control flow op definitions.
def VHLO_AfterAllOpV1 : VHLO_Op<"after_all"> {
  let arguments = (ins Variadic<VHLO_Token>:$inputs);
  let results = (outs VHLO_Token:$result);
}

def VHLO_IfOpV1 : VHLO_Op<"if"> {
  let arguments = (ins VHLO_PredTensor:$pred);
  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_CaseOpV1 : VHLO_Op<"case"> {
  let arguments = (ins I32Tensor:$index);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_WhileOpV1 : VHLO_Op<"while"> {
  let arguments = (ins Variadic<VHLO_TensorOrToken>:$operand);
  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_AllGatherOpV1 : VHLO_Op<"all_gather", "0.3.0", "0.3.0"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_AllGatherOpV2 : VHLO_Op<"all_gather_v2", "0.4.0"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_AllReduceOpV1 : VHLO_Op<"all_reduce"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VHLO_ReduceScatterOpV1 : VHLO_Op<"reduce_scatter"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$scatter_dimension,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VHLO_AllToAllOpV1 : VHLO_Op<"all_to_all"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$split_dimension,
    I64Attr:$concat_dimension,
    I64Attr:$split_count,
    I64ElementsAttr:$replica_groups
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_ReduceOpV1 : VHLO_Op<"reduce", "0.3.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    Variadic<VHLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );
  let results = (outs Variadic<VHLO_Tensor>);
  let regions = (region SizedRegion<1>:$body);
}

//===----------------------------------------------------------------------===//
// VHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def VHLO_GetTupleElementOpV1 : VHLO_Op<"get_tuple_element"> {
  let arguments = (ins
    VHLO_Tuple:$operand,
    I32Attr:$index
  );
  let results = (outs VHLO_TensorOrTokenOrTuple);
}

def VHLO_TupleOpV1 : VHLO_Op<"tuple"> {
  let arguments = (ins Variadic<VHLO_TensorOrTokenOrTuple>:$val);
  let results = (outs VHLO_Tuple:$result);
}

def VHLO_CompareOpV1 : VHLO_Op<"compare"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VHLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<VHLO_ComparisonTypeAttr>:$compare_type
  );
  let results = (outs VHLO_PredTensor);
}

// Slice ops
def VHLO_SliceOpV1 : VHLO_Op<"slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_DynamicSliceOpV1 : VHLO_Op<"dynamic_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    Variadic<VHLO_ScalarIntTensor>:$start_indices,
    I64ElementsAttr:$slice_sizes
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_DynamicUpdateSliceOpV1 : VHLO_Op<"dynamic_update_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$update,
    Variadic<VHLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs VHLO_Tensor:$result);
}

// Other op definitions.
def VHLO_BatchNormGradOpV1 : VHLO_Op<"batch_norm_grad"> {
  let arguments = (ins
    RankedTensorOf<[VHLO_Float]>:$operand,
    1DTensorOf<[VHLO_Float]>:$scale,
    1DTensorOf<[VHLO_Float]>:$mean,
    1DTensorOf<[VHLO_Float]>:$variance,
    RankedTensorOf<[VHLO_Float]>:$grad_output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
  let results = (outs
      RankedTensorOf<[VHLO_Float]>:$grad_operand,
      1DTensorOf<[VHLO_Float]>:$grad_scale,
      1DTensorOf<[VHLO_Float]>:$grad_offset);
}

def VHLO_BatchNormInferenceOpV1 : VHLO_Op<"batch_norm_inference"> {
  let arguments = (ins
    RankedTensorOf<[VHLO_Float]>:$operand,
    1DTensorOf<[VHLO_Float]>:$scale,
    1DTensorOf<[VHLO_Float]>:$offset,
    1DTensorOf<[VHLO_Float]>:$mean,
    1DTensorOf<[VHLO_Float]>:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
  let results = (outs RankedTensorOf<[VHLO_Float]>:$result);
}

def VHLO_BatchNormTrainingOpV1 : VHLO_Op<"batch_norm_training"> {
  let arguments = (ins
    RankedTensorOf<[VHLO_Float]>:$operand,
    1DTensorOf<[VHLO_Float]>:$scale,
    1DTensorOf<[VHLO_Float]>:$offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );
  let results = (outs
      RankedTensorOf<[VHLO_Float]>:$output,
      1DTensorOf<[VHLO_Float]>:$batch_mean,
      1DTensorOf<[VHLO_Float]>:$batch_var);
}

def VHLO_BitcastConvertOpV1 : VHLO_Op<"bitcast_convert"> {
  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_Tensor);
}

def VHLO_BroadcastOpV1 : VHLO_Op<"broadcast"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_BroadcastInDimOpV1 : VHLO_Op<"broadcast_in_dim"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );
  let results = (outs VHLO_StaticShapeTensor);
}

def VHLO_DynamicBroadcastInDimOpV1 : VHLO_Op<"dynamic_broadcast_in_dim"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_DimensionTensor:$output_dimensions,
    BroadcastDimAttr:$broadcast_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_expanding_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_nonexpanding_dimensions
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_CholeskyOpV1 : VHLO_Op<"cholesky"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$lower
  );
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_ClampOpV1 : VHLO_Op<"clamp"> {
  let arguments = (ins
    VHLO_Tensor:$min,
    VHLO_Tensor:$operand,
    VHLO_Tensor:$max
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ConcatenateOpV1 : VHLO_Op<"concatenate"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    I64Attr:$dimension
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_CollectivePermuteOpV1 : VHLO_Op<"collective_permute", "0.3.0", "0.3.0"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$source_target_pairs
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_CollectivePermuteOpV2 : VHLO_Op<"collective_permute_v2", "0.4.0"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$source_target_pairs,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_ConvolutionOpV1 : VHLO_Op<"convolution"> {
  let arguments = !con(
    (ins
       VHLO_Tensor:$lhs,
       VHLO_Tensor:$rhs),
    VHLO_ConvolutionAttributes.attributes);
  let results = (outs VHLO_Tensor);
}

def VHLO_CrossReplicaSumOpV1 : VHLO_Op<"cross-replica-sum"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_CustomCallOpV1 : VHLO_Op<"custom_call", "0.3.0", "0.3.0"> {
  let arguments = (ins
    Variadic<VHLO_TensorOrTokenOrTuple>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    DefaultValuedOptionalAttr<
        VHLO_CustomCallApiVersionAttr,
        "::mlir::vhlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
        $api_version,
    DefaultValuedOptionalAttr<VHLO_FlatSymbolRefArrayAttr, "{}">:$called_computations,
    OptionalAttr<VHLO_ArrayOfLayoutAttr>:$operand_layouts,
    OptionalAttr<VHLO_ArrayOfLayoutAttr>:$result_layouts
  );
  let results = (outs Variadic<VHLO_TensorOrTokenOrTuple>);
}

def VHLO_CustomCallOpV2: VHLO_Op<"custom_call_v2", "0.4.0"> {
  let arguments = (ins
    Variadic<VHLO_TensorOrTokenOrTuple>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    DefaultValuedOptionalAttr<
        VHLO_CustomCallApiVersionAttr,
        "::mlir::vhlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
        $api_version,
    DefaultValuedOptionalAttr<VHLO_FlatSymbolRefArrayAttr, "{}">:$called_computations,
    OptionalAttr<VHLO_ArrayOfLayoutAttr>:$operand_layouts,
    OptionalAttr<VHLO_ArrayOfLayoutAttr>:$result_layouts,
    DefaultValuedOptionalAttr<
        TypedArrayAttrBase<
            VHLO_OutputOperandAlias,
            "Aliasing attribute for outputs and operands of CustomCall">,
        "{}">:$output_operand_aliases
  );
  let results = (outs Variadic<VHLO_TensorOrTokenOrTuple>);
}

def VHLO_DotOpV1 : VHLO_Op<"dot"> {
  let arguments = (
    ins VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_DotGeneralOpV1 : VHLO_Op<"dot_general"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VHLO_DotDimensionNumbers:$dot_dimension_numbers,
    VHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_EinsumOpV1 : VHLO_Op<"einsum"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_UnaryEinsumOpV1 : VHLO_Op<"unary_einsum"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    StrAttr:$einsum_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_FftOpV1 : VHLO_Op<"fft"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$operand,
    VHLO_FftTypeAttr:$fft_type,
    I64ElementsAttr:$fft_length
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_GatherOpV1 : VHLO_Op<"gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_IntTensor:$start_indices,
    VHLO_GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_GetDimensionSizeOpV1 : VHLO_Op<"get_dimension_size"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$dimension
  );
  let results = (outs I32Tensor);
}

def VHLO_MapOpV1 : VHLO_Op<"map"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VHLO_ReshapeOpV1 : VHLO_Op<"reshape"> {
  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_StaticShapeTensor);
}

def VHLO_DynamicReshapeOpV1 : VHLO_Op<"dynamic_reshape"> {
  let arguments = (ins VHLO_Tensor:$operand, VHLO_DimensionTensor:$output_shape);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ScatterOpV1 : VHLO_Op<"scatter", "0.3.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    TensorOf<[AnyInteger, Index]>:$scatter_indices,
    Variadic<VHLO_Tensor>:$updates,
    VHLO_ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unique_indices
  );
  let regions = (region SizedRegion<1>:$update_computation);
  let results = (outs Variadic<VHLO_Tensor>);
}

def VHLO_SelectOpV1 : VHLO_Op<"select"> {
  let arguments = (ins
    VHLO_PredTensor:$pred,
    VHLO_Tensor:$on_true,
    VHLO_Tensor:$on_false
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_SelectAndScatterOpV1 : VHLO_Op<"select_and_scatter"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$source,
    VHLO_Tensor:$init_value,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );
  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);
  let results = (outs VHLO_Tensor);
}

def VHLO_SetDimensionSizeOpV1 : VHLO_Op<"set_dimension_size"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I32Tensor:$size,
    I64Attr:$dimension
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_SortOpV1 : VHLO_Op<"sort"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable
  );
  let regions = (region SizedRegion<1>:$comparator);
  let results = (outs Variadic<VHLO_Tensor>);
}

def VHLO_ReverseOpV1 : VHLO_Op<"reverse"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_PadOpV1 : VHLO_Op<"pad"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high,
    I64ElementsAttr:$interior_padding
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_TraceOpV1 : VHLO_Op<"trace"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    StrAttr:$tag
  );
}

def VHLO_TransposeOpV1 : VHLO_Op<"transpose"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_TriangularSolveOpV1 : VHLO_Op<"triangular_solve"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$a,
    VHLO_FpOrComplexTensor:$b,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    VHLO_TransposeAttr:$transpose_a
  );
  let results = (outs VHLO_FpOrComplexTensor);
}

def VHLO_ReduceWindowOpV1 : VHLO_Op<"reduce_window", "0.3.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    Variadic<VHLO_Tensor>:$init_values,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the operand dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );
  let results = (outs Variadic<VHLO_Tensor>);
  let regions = (region SizedRegion<1>:$body);
}

def VHLO_TorchIndexSelectOpV1 : VHLO_Op<"torch_index_select"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$index,
    I64Attr:$dim,
    I64Attr:$batch_dims
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_OptimizationBarrierOpV1 : VHLO_Op<"optimization_barrier"> {
  let arguments = (ins Variadic<VHLO_TensorOrToken>:$operand);
  let results = (outs Variadic<VHLO_TensorOrToken>:$result);
}

//===----------------------------------------------------------------------===//
// VHLO RNG Operators.
//===----------------------------------------------------------------------===//

def VHLO_RngOpV1 : VHLO_Op<"rng"> {
  let arguments = (ins
    0DTensorOf<[VHLO_Pred, VHLO_Int, VHLO_Float]>:$a,
    0DTensorOf<[VHLO_Pred, VHLO_Int, VHLO_Float]>:$b,
    VHLO_DimensionTensor:$shape,
    VHLO_RngDistributionAttr:$rng_distribution
  );
  let results = (outs VHLO_PredIntOrFpTensor:$result);
}

def VHLO_RngBitGeneratorOpV1 : VHLO_Op<"rng_bit_generator"> {
  let arguments = (ins
    VHLO_RngAlgorithmAttr:$rng_algorithm,
    VHLO_IntOrFpTensor:$initial_state
  );
  let results = (outs
    VHLO_IntOrFpTensor:$output_state,
    VHLO_IntOrFpTensor:$output
  );
}

// Quantize Ops
def VHLO_UniformQuantizeOpV1 : VHLO_Op<"uniform_quantize"> {
  let arguments = (ins TensorOf<[F32, BF16, VHLO_QuantizedInt]>:$operand);
  let results = (outs VHLO_QuantizedIntTensor:$result);
}

def VHLO_UniformDequantizeOpV1 : VHLO_Op<"uniform_dequantize"> {
  let arguments = (ins VHLO_QuantizedIntTensor:$operand);
  let results = (outs TensorOf<[F32, BF16]>:$result);
}

def VHLO_ReducePrecisionOpV1 : VHLO_Op<"reduce_precision"> {
  let arguments = (ins
    VHLO_FpTensor:$operand,
    I32Attr:$exponent_bits,
    I32Attr:$mantissa_bits
  );
  let results = (outs VHLO_FpTensor:$output);
}

def VHLO_RealDynamicSliceOpV1 : VHLO_Op<"real_dynamic_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_DimensionTensor:$start_indices,
    VHLO_DimensionTensor:$limit_indices,
    VHLO_DimensionTensor:$strides
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_DynamicPadOpV1 : VHLO_Op<"dynamic_pad"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$padding_value,
    VHLO_DimensionTensor:$edge_padding_low,
    VHLO_DimensionTensor:$edge_padding_high,
    VHLO_DimensionTensor:$interior_padding
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_DynamicGatherOpV1 : VHLO_Op<"dynamic_gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_IntTensor:$start_indices,
    VHLO_IntTensor:$slice_sizes,
    VHLO_GatherDimensionNumbers:$dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_DynamicConvOpV1 : VHLO_Op<"dynamic_conv"> {
  let arguments = !con(
    (ins
       VHLO_Tensor:$lhs,
       VHLO_Tensor:$rhs,
       VHLO_Tensor:$d_padding),
    VHLO_ConvolutionAttributes.attributes);
  let results = (outs VHLO_Tensor);
}

def VHLO_ComputeReshapeShapeOpV1 : VHLO_Op<"compute_reshape_shape"> {
  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs 1DTensorOf<[AnyInteger, Index]>:$result);
}

def VHLO_CstrReshapableOpV1 : VHLO_Op<"cstr_reshapable"> {
  let results = (outs Shape_WitnessType:$result);
  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
}

#endif // STABLEHLO_DIALECT_STABLEHLO_OPS
