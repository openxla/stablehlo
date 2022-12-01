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

#ifndef STABLEHLO_COMPATIBILITY_VHLO_OPS
#define STABLEHLO_COMPATIBILITY_VHLO_OPS

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
      0.0.0: Base release.
      0.1.0: CustomCallOp add result aliasing.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

include "stablehlo/dialect/VhloEnums.td"
include "stablehlo/dialect/VhloAttrs.td"

// Only use this base class if necessary.
// - ReturnOp needs a trait for Terminator.
// - ReduceOp/ReduceWindowOp/ScatterOp need a trait since they have
//   multiple variadic arguments.
class VHLO_OpWithTraits<string mnemonic, list<Trait> traits, string minVersion = "0.0.0", string maxVersion = "current"> :
    Op<VHLO_Dialect, mnemonic, 
      [DeclareOpInterfaceMethods<VersionedInterface>] # traits> {
  let extraClassDefinition = [{
    mlir::vhlo::Version $cppClass::getMinVersion() {
      return *mlir::vhlo::Version::get("}] #  minVersion # [{");
    }
    mlir::vhlo::Version $cppClass::getMaxVersion() {
      if (!strcmp("}] # maxVersion # [{", "current")) return VhloDialect::getCurrentDialectVersion();
      return *mlir::vhlo::Version::get("}] #  maxVersion # [{");
    }
  }];
}

class VHLO_Op<string mnemonic, string minVersion = "0.0.0", string maxVersion = "current"> :
    VHLO_OpWithTraits<mnemonic, [], minVersion, maxVersion>;

def VHLO_ConstantOp : VHLO_Op<"constant"> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs VHLO_StaticShapeTensor:$output);
}

def VHLO_CreateTokenOp : VHLO_Op<"create_token"> {
  let results = (outs VHLO_Token:$output);
}

def VHLO_IotaOp : VHLO_Op<"iota"> {
  let arguments = (ins I64Attr:$iota_dimension);
  let results = (outs VHLO_IntFpOrComplexTensor:$output);
}

def VHLO_DynamicIotaOp: VHLO_Op<"dynamic_iota"> {
  let arguments = (ins VHLO_DimensionTensor:$output_shape, I64Attr:$iota_dimension);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_AbsOp: VHLO_Op<"abs"> {
  let arguments = (ins TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$operand);
  let results = (outs TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$result);
}

def VHLO_CbrtOp: VHLO_Op<"cbrt"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_CeilOp: VHLO_Op<"ceil"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_ConvertOp : VHLO_Op<"convert"> {
  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ClzOp: VHLO_Op<"count_leading_zeros"> {
  let arguments = (ins VHLO_IntTensor:$operand);
  let results = (outs VHLO_IntTensor:$result);
}

def VHLO_CosineOp: VHLO_Op<"cosine"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_ExpOp: VHLO_Op<"exponential"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_Expm1Op: VHLO_Op<"exponential_minus_one"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}
def VHLO_FloorOp: VHLO_Op<"floor"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_ImagOp: VHLO_Op<"imag"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_IsFiniteOp: VHLO_Op<"is_finite"> {
  let arguments = (ins VHLO_FpTensor:$x);
  let results = (outs VHLO_PredTensor:$y);
}

def VHLO_LogOp: VHLO_Op<"log">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_Log1pOp: VHLO_Op<"log_plus_one">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_LogisticOp: VHLO_Op<"logistic">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_NotOp: VHLO_Op<"not">{
  let arguments = (ins VHLO_PredOrIntTensor:$operand);
  let results = (outs VHLO_PredOrIntTensor:$result);
}

def VHLO_NegOp: VHLO_Op<"negate">{
  let arguments = (ins VHLO_IntFpOrComplexTensor:$operand);
  let results = (outs VHLO_IntFpOrComplexTensor:$result);
}

def VHLO_PopulationCountOp: VHLO_Op<"popcnt">{
  let arguments = (ins VHLO_IntTensor:$operand);
  let results = (outs VHLO_IntTensor:$result);
}

def VHLO_RealOp: VHLO_Op<"real">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_RoundOp: VHLO_Op<"round_nearest_afz">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_RoundNearestEvenOp: VHLO_Op<"round_nearest_even">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VHLO_RsqrtOp: VHLO_Op<"rsqrt">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_SignOp: VHLO_Op<"sign">{
  let arguments = (ins TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$operand);
  let results = (outs TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$result);
}

def VHLO_SineOp: VHLO_Op<"sine">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_SqrtOp: VHLO_Op<"sqrt">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_TanhOp: VHLO_Op<"tanh">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

// Binary Ops
def VHLO_AddOp : VHLO_Op<"add"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_Atan2Op : VHLO_Op<"atan2"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ComplexOp: VHLO_Op<"complex"> {
  let arguments = (ins VHLO_Fp32Or64Tensor:$lhs, VHLO_Fp32Or64Tensor:$rhs);
  let results = (outs VHLO_ComplexTensor:$result);
}
def VHLO_DivOp : VHLO_Op<"divide"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_MaxOp : VHLO_Op<"maximum"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_MinOp : VHLO_Op<"minimum"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_MulOp : VHLO_Op<"multiply"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_PowOp : VHLO_Op<"power"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_RemOp : VHLO_Op<"remainder"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ShiftLeftOp : VHLO_Op<"shift_left"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ShiftRightArithmeticOp : VHLO_Op<"shift_right_arithmetic"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_ShiftRightLogicalOp : VHLO_Op<"shift_right_logical"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_SubtractOp : VHLO_Op<"subtract"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}

// Logical Ops
def VHLO_AndOp: VHLO_Op<"and"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_OrOp: VHLO_Op<"or"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VHLO_XorOp : VHLO_Op<"xor"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ReturnOp : VHLO_OpWithTraits<"return", [Terminator]> {
  let arguments = (ins
    Variadic<VHLO_TensorOrTokenOrTuple >:$results
  );
  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

// Communication op definitions.
def VHLO_InfeedOp : VHLO_Op<"infeed"> {
  let arguments = (ins
    VHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$infeed_config,
    OptionalAttr<ArrayAttr>:$layout
  );
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_OutfeedOp : VHLO_Op<"outfeed"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    VHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$outfeed_config
  );
  let results = (outs VHLO_Token);
}

def VHLO_SendOp : VHLO_Op<"send"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    VHLO_Token:$token,
    VHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );
  let results = (outs VHLO_Token);
}

def VHLO_RecvOp : VHLO_Op<"recv"> {
  let arguments = (ins
    VHLO_Token:$token,
    VHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

// Parallelism related op definitions.
def VHLO_ReplicaIdOp : VHLO_Op<"replica_id"> {
  let results = (outs TensorOf<[UI32]>);
}

// Control flow op definitions.
def VHLO_AfterAllOp : VHLO_Op<"after_all"> {
  let arguments = (ins Variadic<VHLO_Token>:$inputs);
  let results = (outs VHLO_Token:$result);
}

def VHLO_IfOp: VHLO_Op<"if"> {
  let arguments = (ins VHLO_PredTensor:$pred);
  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_CaseOp : VHLO_Op<"case"> {
  let arguments = (ins I32Tensor:$index);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_WhileOp: VHLO_Op<"while"> {
  let arguments = (ins Variadic<VHLO_TensorOrToken>:$operand);
  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VHLO_AllGatherOp : VHLO_Op<"all_gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_AllReduceOp : VHLO_Op<"all_reduce"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VHLO_ReduceScatterOp : VHLO_Op<"reduce_scatter"> {
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

def VHLO_AllToAllOp : VHLO_Op<"all_to_all"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$split_dimension,
    I64Attr:$concat_dimension,
    I64Attr:$split_count,
    I64ElementsAttr:$replica_groups
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_ReduceOp: VHLO_OpWithTraits<"reduce", [SameVariadicOperandSize]> {
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
def VHLO_GetTupleElementOp: VHLO_Op<"get_tuple_element"> {
  let arguments = (ins
    VHLO_Tuple:$operand,
    I32Attr:$index
  );
  let results = (outs VHLO_TensorOrTokenOrTuple);
}

def VHLO_TupleOp : VHLO_Op<"tuple"> {
  let arguments = (ins Variadic<VHLO_TensorOrTokenOrTuple>:$val);
  let results = (outs VHLO_Tuple:$result);
}

def VHLO_CompareOp: VHLO_Op<"compare"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VHLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<VHLO_ComparisonTypeAttr>:$compare_type
  );
  let results = (outs VHLO_PredTensor);
}

// Slice ops
def VHLO_SliceOp: VHLO_Op<"slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_DynamicSliceOp: VHLO_Op<"dynamic_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    Variadic<VHLO_ScalarIntTensor>:$start_indices,
    I64ElementsAttr:$slice_sizes
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_DynamicUpdateSliceOp: VHLO_Op<"dynamic_update_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$update,
    Variadic<VHLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs VHLO_Tensor:$result);
}

// Other op definitions.
def VHLO_BatchNormGradOp : VHLO_Op<"batch_norm_grad"> {
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

def VHLO_BatchNormInferenceOp : VHLO_Op<"batch_norm_inference"> {
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

def VHLO_BatchNormTrainingOp : VHLO_Op<"batch_norm_training"> {
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

def VHLO_BitcastConvertOp : VHLO_Op<"bitcast_convert"> {
  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_Tensor);
}

def VHLO_BroadcastOp : VHLO_Op<"broadcast"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_BroadcastInDimOp : VHLO_Op<"broadcast_in_dim"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );
  let results = (outs VHLO_StaticShapeTensor);
}

def VHLO_DynamicBroadcastInDimOp : VHLO_Op<"dynamic_broadcast_in_dim"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_DimensionTensor:$output_dimensions,
    BroadcastDimAttr:$broadcast_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_expanding_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_nonexpanding_dimensions
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_CholeskyOp : VHLO_Op<"cholesky"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$lower
  );
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VHLO_ClampOp : VHLO_Op<"clamp"> {
  let arguments = (ins
    VHLO_Tensor:$min,
    VHLO_Tensor:$operand,
    VHLO_Tensor:$max
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ConcatenateOp : VHLO_Op<"concatenate"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    I64Attr:$dimension
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_CollectivePermuteOp: VHLO_Op<"collective_permute"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$source_target_pairs,
    OptionalAttr<VHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_ConvolutionOp : VHLO_Op<"convolution"> {
  let arguments = !con(
    (ins
       VHLO_Tensor:$lhs,
       VHLO_Tensor:$rhs),
    VHLO_ConvolutionAttributes.attributes);
  let results = (outs VHLO_Tensor);
}

def VHLO_CrossReplicaSumOp : VHLO_Op<"cross-replica-sum"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_CustomCallOp: VHLO_Op<"custom_call", "0.0.0", "0.0.0"> {
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
    OptionalAttr<VHLO_ArrayOfLayoutAttr>:$operand_layouts
  );
  let results = (outs Variadic<VHLO_TensorOrTokenOrTuple>);
}

def VHLO_CustomCallOpV2: VHLO_Op<"custom_call_v2", "0.1.0"> {
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

def VHLO_DotOp: VHLO_Op<"dot"> {
  let arguments = (
    ins VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_DotGeneralOp: VHLO_Op<"dot_general"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VHLO_DotDimensionNumbers:$dot_dimension_numbers,
    VHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_EinsumOp: VHLO_Op<"einsum"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_UnaryEinsumOp: VHLO_Op<"unary_einsum"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    StrAttr:$einsum_config
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_FftOp: VHLO_Op<"fft"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$operand,
    VHLO_FftTypeAttr:$fft_type,
    I64ElementsAttr:$fft_length
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_GatherOp: VHLO_Op<"gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_IntTensor:$start_indices,
    VHLO_GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_GetDimensionSizeOp: VHLO_Op<"get_dimension_size"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$dimension
  );
  let results = (outs I32Tensor);
}

def VHLO_MapOp: VHLO_Op<"map"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VHLO_ReshapeOp: VHLO_Op<"reshape"> {
  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_StaticShapeTensor);
}

def VHLO_DynamicReshapeOp: VHLO_Op<"dynamic_reshape"> {
  let arguments = (ins VHLO_Tensor:$operand, VHLO_DimensionTensor:$output_shape);
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_ScatterOp: VHLO_OpWithTraits<"scatter", [SameVariadicOperandSize]> {
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

def VHLO_SelectOp: VHLO_Op<"select"> {
  let arguments = (ins
    VHLO_PredTensor:$pred,
    VHLO_Tensor:$on_true,
    VHLO_Tensor:$on_false
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_SelectAndScatterOp: VHLO_Op<"select_and_scatter"> {
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

def VHLO_SetDimensionSizeOp: VHLO_Op<"set_dimension_size"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I32Tensor:$size,
    I64Attr:$dimension
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_SortOp : VHLO_Op<"sort"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable
  );
  let regions = (region SizedRegion<1>:$comparator);
  let results = (outs Variadic<VHLO_Tensor>);
}

def VHLO_ReverseOp: VHLO_Op<"reverse"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_PadOp: VHLO_Op<"pad"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high,
    I64ElementsAttr:$interior_padding
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_TraceOp: VHLO_Op<"trace"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    StrAttr:$tag
  );
}

def VHLO_TransposeOp: VHLO_Op<"transpose"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_TriangularSolveOp: VHLO_Op<"triangular_solve"> {
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

def VHLO_ReduceWindowOp: VHLO_OpWithTraits<"reduce_window", [SameVariadicOperandSize]> {
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

def VHLO_TorchIndexSelectOp : VHLO_Op<"torch_index_select"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$index,
    I64Attr:$dim,
    I64Attr:$batch_dims
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_OptimizationBarrierOp : VHLO_Op<"optimization_barrier"> {
  let arguments = (ins Variadic<VHLO_TensorOrToken>:$operand);
  let results = (outs Variadic<VHLO_TensorOrToken>:$result);
}

//===----------------------------------------------------------------------===//
// VHLO RNG Operators.
//===----------------------------------------------------------------------===//

def VHLO_RngOp : VHLO_Op<"rng"> {
  let arguments = (ins
    0DTensorOf<[VHLO_Pred, VHLO_Int, VHLO_Float]>:$a,
    0DTensorOf<[VHLO_Pred, VHLO_Int, VHLO_Float]>:$b,
    VHLO_DimensionTensor:$shape,
    VHLO_RngDistributionAttr:$rng_distribution
  );
  let results = (outs VHLO_PredIntOrFpTensor:$result);
}

def VHLO_RngBitGeneratorOp : VHLO_Op<"rng_bit_generator"> {
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
def VHLO_UniformQuantizeOp : VHLO_Op<"uniform_quantize"> {
  let arguments = (ins TensorOf<[F32, BF16, VHLO_QuantizedInt]>:$operand);
  let results = (outs VHLO_QuantizedIntTensor:$result);
}

def VHLO_UniformDequantizeOp : VHLO_Op<"uniform_dequantize"> {
  let arguments = (ins VHLO_QuantizedIntTensor:$operand);
  let results = (outs TensorOf<[F32, BF16]>:$result);
}

def VHLO_ReducePrecisionOp : VHLO_Op<"reduce_precision"> {
  let arguments = (ins
    VHLO_FpTensor:$operand,
    I32Attr:$exponent_bits,
    I32Attr:$mantissa_bits
  );
  let results = (outs VHLO_FpTensor:$output);
}

def VHLO_RealDynamicSliceOp: VHLO_Op<"real_dynamic_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_DimensionTensor:$start_indices,
    VHLO_DimensionTensor:$limit_indices,
    VHLO_DimensionTensor:$strides
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_DynamicPadOp: VHLO_Op<"dynamic_pad"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$padding_value,
    VHLO_DimensionTensor:$edge_padding_low,
    VHLO_DimensionTensor:$edge_padding_high,
    VHLO_DimensionTensor:$interior_padding
  );
  let results = (outs VHLO_Tensor:$result);
}

def VHLO_DynamicGatherOp: VHLO_Op<"dynamic_gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_IntTensor:$start_indices,
    VHLO_IntTensor:$slice_sizes,
    VHLO_GatherDimensionNumbers:$dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs VHLO_Tensor);
}

def VHLO_DynamicConvOp : VHLO_Op<"dynamic_conv"> {
  let arguments = !con(
    (ins
       VHLO_Tensor:$lhs,
       VHLO_Tensor:$rhs,
       VHLO_Tensor:$d_padding),
    VHLO_ConvolutionAttributes.attributes);
  let results = (outs VHLO_Tensor);
}

def VHLO_ComputeReshapeShapeOp : VHLO_Op<"compute_reshape_shape"> {
  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs 1DTensorOf<[AnyInteger, Index]>:$result);
}

def VHLO_CstrReshapableOp : VHLO_Op<"cstr_reshapable"> {
  let results = (outs Shape_WitnessType:$result);
  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
}

#endif // STABLEHLO_COMPATIBILITY_STABLEHLO_OPS