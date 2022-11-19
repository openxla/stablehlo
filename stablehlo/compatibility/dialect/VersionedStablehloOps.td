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

#ifndef STABLEHLO_COMPATIBILITY_VERSIONEDHLO_OPS
#define STABLEHLO_COMPATIBILITY_VERSIONEDHLO_OPS

include "stablehlo/compatibility/dialect/VersionedStablehloEnums.td"
include "stablehlo/compatibility/dialect/VersionedStablehloAttrs.td"

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect and Ops
//===----------------------------------------------------------------------===//

def VersionInterface : OpInterface<"VersionInterface"> {
  let methods = [
    InterfaceMethod<
      "Description...", "mlir::versionedhlo::VersionNumber", "getMinVersion">,
    InterfaceMethod<
      "Description...", "mlir::versionedhlo::VersionNumber", "getMaxVersion">,
  ];
}

// Only use this base class if necessary.
// - ReturnOp needs a trait for Terminator.
// - ReduceOp/ReduceWindowOp/ScatterOp need a trait since they have
//   multiple variadic arguments.
class VersionedHLO_OpWithTraits<string mnemonic, list<Trait> traits, string minVersion = "0.0.0", string maxVersion = "current"> :
    Op<VersionedHLO_Dialect, mnemonic, 
      [DeclareOpInterfaceMethods<VersionInterface>] # traits> {
  let extraClassDefinition = [{
    mlir::versionedhlo::VersionNumber $cppClass::getMinVersion() {
      return *mlir::versionedhlo::VersionNumber::get("}] #  minVersion # [{");
    }
    mlir::versionedhlo::VersionNumber $cppClass::getMaxVersion() {
      return *mlir::versionedhlo::VersionNumber::get("}] #  maxVersion # [{");
    }
  }];
}

class VersionedHLO_Op<string mnemonic, string minVersion = "0.0.0", string maxVersion = "current"> :
    VersionedHLO_OpWithTraits<mnemonic, [], minVersion, maxVersion>;

def VersionedHLO_ConstantOp : VersionedHLO_Op<"constant"> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs VHLO_StaticShapeTensor:$output);
}

def VersionedHLO_CreateTokenOp : VersionedHLO_Op<"create_token"> {
  let results = (outs VHLO_Token:$output);
}

def VersionedHLO_IotaOp : VersionedHLO_Op<"iota"> {
  let arguments = (ins I64Attr:$iota_dimension);
  let results = (outs VHLO_IntFpOrComplexTensor:$output);
}

def VersionedHLO_DynamicIotaOp: VersionedHLO_Op<"dynamic_iota"> {
  let arguments = (ins VHLO_DimensionTensor:$output_shape, I64Attr:$iota_dimension);
  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_AbsOp: VersionedHLO_Op<"abs"> {
  let arguments = (ins TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$operand);
  let results = (outs TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$result);
}

def VersionedHLO_CbrtOp: VersionedHLO_Op<"cbrt"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VersionedHLO_CeilOp: VersionedHLO_Op<"ceil"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VersionedHLO_ConvertOp : VersionedHLO_Op<"convert"> {
  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_ClzOp: VersionedHLO_Op<"count_leading_zeros"> {
  let arguments = (ins VHLO_IntTensor:$operand);
  let results = (outs VHLO_IntTensor:$result);
}

def VersionedHLO_CosineOp: VersionedHLO_Op<"cosine"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_ExpOp: VersionedHLO_Op<"exponential"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_Expm1Op: VersionedHLO_Op<"exponential_minus_one"> {
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}
def VersionedHLO_FloorOp: VersionedHLO_Op<"floor"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VersionedHLO_ImagOp: VersionedHLO_Op<"imag"> {
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VersionedHLO_IsFiniteOp: VersionedHLO_Op<"is_finite"> {
  let arguments = (ins VHLO_FpTensor:$x);
  let results = (outs VHLO_PredTensor:$y);
}

def VersionedHLO_LogOp: VersionedHLO_Op<"log">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_Log1pOp: VersionedHLO_Op<"log_plus_one">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_LogisticOp: VersionedHLO_Op<"logistic">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_NotOp: VersionedHLO_Op<"not">{
  let arguments = (ins VHLO_PredOrIntTensor:$operand);
  let results = (outs VHLO_PredOrIntTensor:$result);
}

def VersionedHLO_NegOp: VersionedHLO_Op<"negate">{
  let arguments = (ins VHLO_IntFpOrComplexTensor:$operand);
  let results = (outs VHLO_IntFpOrComplexTensor:$result);
}

def VersionedHLO_PopulationCountOp: VersionedHLO_Op<"popcnt">{
  let arguments = (ins VHLO_IntTensor:$operand);
  let results = (outs VHLO_IntTensor:$result);
}

def VersionedHLO_RealOp: VersionedHLO_Op<"real">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VersionedHLO_RoundOp: VersionedHLO_Op<"round_nearest_afz">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VersionedHLO_RoundNearestEvenOp: VersionedHLO_Op<"round_nearest_even">{
  let arguments = (ins VHLO_FpTensor:$operand);
  let results = (outs VHLO_FpTensor:$result);
}

def VersionedHLO_RsqrtOp: VersionedHLO_Op<"rsqrt">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_SignOp: VersionedHLO_Op<"sign">{
  let arguments = (ins TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$operand);
  let results = (outs TensorOf<[VHLO_SInt, VHLO_Float, VHLO_Complex]>:$result);
}

def VersionedHLO_SineOp: VersionedHLO_Op<"sine">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_SqrtOp: VersionedHLO_Op<"sqrt">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_TanhOp: VersionedHLO_Op<"tanh">{
  let arguments = (ins VHLO_FpOrComplexTensor:$operand);
  let results = (outs VHLO_FpOrComplexTensor:$result);
}

// Binary Ops
def VersionedHLO_AddOp : VersionedHLO_Op<"add"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_Atan2Op : VersionedHLO_Op<"atan2"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_ComplexOp: VersionedHLO_Op<"complex"> {
  let arguments = (ins VHLO_Fp32Or64Tensor:$lhs, VHLO_Fp32Or64Tensor:$rhs);
  let results = (outs VHLO_ComplexTensor:$result);
}
def VersionedHLO_DivOp : VersionedHLO_Op<"divide"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_MaxOp : VersionedHLO_Op<"maximum"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_MinOp : VersionedHLO_Op<"minimum"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_MulOp : VersionedHLO_Op<"multiply"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_PowOp : VersionedHLO_Op<"power"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_RemOp : VersionedHLO_Op<"remainder"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_ShiftLeftOp : VersionedHLO_Op<"shift_left"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_ShiftRightArithmeticOp : VersionedHLO_Op<"shift_right_arithmetic"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_ShiftRightLogicalOp : VersionedHLO_Op<"shift_right_logical"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_SubtractOp : VersionedHLO_Op<"subtract"> {
  let arguments = (ins VHLO_Tensor:$lhs, VHLO_Tensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}

// Logical Ops
def VersionedHLO_AndOp: VersionedHLO_Op<"and"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_OrOp: VersionedHLO_Op<"or"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}
def VersionedHLO_XorOp : VersionedHLO_Op<"xor"> {
  let arguments = (ins VHLO_PredOrIntTensor:$lhs, VHLO_PredOrIntTensor:$rhs);
  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_ReturnOp : VersionedHLO_OpWithTraits<"return", [Terminator]> {
  let arguments = (ins
    Variadic<VHLO_TensorOrTokenOrTuple >:$results
  );

  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

///// SPLIT //////


// Communication op definitions.
def VersionedHLO_InfeedOp : VersionedHLO_Op<"infeed"> {
  let arguments = (ins
    VHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$infeed_config,
    OptionalAttr<ArrayAttr>:$layout
  );
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VersionedHLO_OutfeedOp : VersionedHLO_Op<"outfeed"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    VHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$outfeed_config
  );
  let results = (outs VHLO_Token);
}

def VersionedHLO_SendOp : VersionedHLO_Op<"send"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    VHLO_Token:$token,
    VersionedHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs VHLO_Token);
}

def VersionedHLO_RecvOp : VersionedHLO_Op<"recv"> {
  let arguments = (ins
    VHLO_Token:$token,
    VersionedHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs Variadic<VHLO_TensorOrToken>);
}

// Parallelism related op definitions.
def VersionedHLO_ReplicaIdOp : VersionedHLO_Op<"replica_id"> {
  let results = (outs TensorOf<[UI32]>);
}

// Control flow op definitions.
def VersionedHLO_AfterAllOp : VersionedHLO_Op<"after_all"> {
  let arguments = (ins Variadic<VHLO_Token>:$inputs);
  let results = (outs VHLO_Token:$result);
}

def VersionedHLO_IfOp: VersionedHLO_Op<"if"> {
  let arguments = (ins VHLO_PredTensor:$pred);

  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);

  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VersionedHLO_CaseOp : VersionedHLO_Op<"case"> {
  let arguments = (ins I32Tensor:$index);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VersionedHLO_WhileOp: VersionedHLO_Op<"while"> {
  let arguments = (ins Variadic<VHLO_TensorOrToken>:$operand);
  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);
  let results = (outs Variadic<VHLO_TensorOrToken>);
}

def VersionedHLO_AllGatherOp : VersionedHLO_Op<"all_gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VersionedHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_AllReduceOp : VersionedHLO_Op<"all_reduce"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VersionedHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_ReduceScatterOp : VersionedHLO_Op<"reduce_scatter"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$scatter_dimension,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<VersionedHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_AllToAllOp : VersionedHLO_Op<"all_to_all"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$split_dimension,
    I64Attr:$concat_dimension,
    I64Attr:$split_count,
    I64ElementsAttr:$replica_groups
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_ReduceOp: VersionedHLO_OpWithTraits<"reduce", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    Variadic<VHLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );

  let results = (outs Variadic<VHLO_Tensor>);

  let regions = (region SizedRegion<1>:$body);
}

//===----------------------------------------------------------------------===//
// VersionedHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def VersionedHLO_GetTupleElementOp: VersionedHLO_Op<"get_tuple_element"> {
  let arguments = (ins
    VHLO_Tuple:$operand,
    I32Attr:$index
  );

  let results = (outs VHLO_TensorOrTokenOrTuple);
}

def VersionedHLO_TupleOp : VersionedHLO_Op<"tuple"> {
  let arguments = (ins Variadic<VHLO_TensorOrTokenOrTuple>:$val);
  let results = (outs VHLO_Tuple:$result);
}

def VersionedHLO_CompareOp: VersionedHLO_Op<"compare"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VersionedHLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<VersionedHLO_ComparisonTypeAttr>:$compare_type
  );
  let results = (outs VHLO_PredTensor);
}

// Slice ops
def VersionedHLO_SliceOp: VersionedHLO_Op<"slice"> {

  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_DynamicSliceOp: VersionedHLO_Op<"dynamic_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    Variadic<VHLO_ScalarIntTensor>:$start_indices,
    I64ElementsAttr:$slice_sizes
  );

  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_DynamicUpdateSliceOp: VersionedHLO_Op<"dynamic_update_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$update,
    Variadic<VHLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs VHLO_Tensor:$result);
}

// Other op definitions.
def VersionedHLO_BatchNormGradOp : VersionedHLO_Op<"batch_norm_grad"> {

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

def VersionedHLO_BatchNormInferenceOp : VersionedHLO_Op<"batch_norm_inference"> {

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

def VersionedHLO_BatchNormTrainingOp : VersionedHLO_Op<"batch_norm_training"> {

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

def VersionedHLO_BitcastConvertOp : VersionedHLO_Op<"bitcast_convert"> {

  let arguments = (ins VHLO_Tensor:$operand);
  let results = (outs VHLO_Tensor);
  
}

def VersionedHLO_BroadcastOp : VersionedHLO_Op<"broadcast"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );

  let results = (outs VHLO_Tensor);

  
}

def VersionedHLO_BroadcastInDimOp : VersionedHLO_Op<"broadcast_in_dim"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs VHLO_StaticShapeTensor);
}

def VersionedHLO_DynamicBroadcastInDimOp : VersionedHLO_Op<"dynamic_broadcast_in_dim"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_DimensionTensor:$output_dimensions,
    BroadcastDimAttr:$broadcast_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_expanding_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_nonexpanding_dimensions
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_CholeskyOp : VersionedHLO_Op<"cholesky"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$lower
  );

  let results = (outs VHLO_FpOrComplexTensor:$result);
}

def VersionedHLO_ClampOp : VersionedHLO_Op<"clamp"> {
  let arguments = (ins
    VHLO_Tensor:$min,
    VHLO_Tensor:$operand,
    VHLO_Tensor:$max
  );
  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_ConcatenateOp : VersionedHLO_Op<"concatenate"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    I64Attr:$dimension
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_CollectivePermuteOp: VersionedHLO_Op<"collective_permute"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$source_target_pairs,
    OptionalAttr<VersionedHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_ConvolutionOp : VersionedHLO_Op<"convolution"> {
  let arguments = !con(
    (ins
       VHLO_Tensor:$lhs,
       VHLO_Tensor:$rhs),
    VersionedHLO_ConvolutionAttributes.attributes);

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_CrossReplicaSumOp : VersionedHLO_Op<"cross-replica-sum"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_CustomCallOp: VersionedHLO_Op<"custom_call", "0.0.0", "0.0.0"> {
  let arguments = (ins
    Variadic<VHLO_TensorOrTokenOrTuple>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    DefaultValuedOptionalAttr<
        VersionedHLO_CustomCallApiVersionAttr,
        "::mlir::versionedhlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
        $api_version,
    DefaultValuedOptionalAttr<VersionedHLO_FlatSymbolRefArrayAttr, "{}">:$called_computations,
    OptionalAttr<VersionedHLO_ArrayOfLayoutAttr>:$operand_layouts
  );
  let results = (outs Variadic<VHLO_TensorOrTokenOrTuple>);
}

def VersionedHLO_CustomCallOpV2: VersionedHLO_Op<"custom_call_v2", "0.1.0"> {
  let arguments = (ins
    Variadic<VHLO_TensorOrTokenOrTuple>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    DefaultValuedOptionalAttr<
        VersionedHLO_CustomCallApiVersionAttr,
        "::mlir::versionedhlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
        $api_version,
    DefaultValuedOptionalAttr<VersionedHLO_FlatSymbolRefArrayAttr, "{}">:$called_computations,
    OptionalAttr<VersionedHLO_ArrayOfLayoutAttr>:$operand_layouts,
    OptionalAttr<VersionedHLO_ArrayOfLayoutAttr>:$result_layouts
  );
  let results = (outs Variadic<VHLO_TensorOrTokenOrTuple>);
}

def VersionedHLO_DotOp: VersionedHLO_Op<"dot"> {
  let arguments = (
    ins VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VersionedHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_DotGeneralOp: VersionedHLO_Op<"dot_general"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    VersionedHLO_DotDimensionNumbers:$dot_dimension_numbers,
    VersionedHLO_PrecisionConfigAttr:$precision_config
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_EinsumOp: VersionedHLO_Op<"einsum"> {
  let arguments = (ins
    VHLO_Tensor:$lhs,
    VHLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_UnaryEinsumOp: VersionedHLO_Op<"unary_einsum"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    StrAttr:$einsum_config
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_FftOp: VersionedHLO_Op<"fft"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$operand,
    VersionedHLO_FftTypeAttr:$fft_type,
    I64ElementsAttr:$fft_length
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_GatherOp: VersionedHLO_Op<"gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_IntTensor:$start_indices,
    VersionedHLO_GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_GetDimensionSizeOp: VersionedHLO_Op<"get_dimension_size"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64Attr:$dimension
  );
  let results = (outs I32Tensor);
}

def VersionedHLO_MapOp: VersionedHLO_Op<"map"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_ReshapeOp: VersionedHLO_Op<"reshape"> {
  let arguments = (ins VHLO_Tensor:$operand);

  let results = (outs VHLO_StaticShapeTensor);
}

def VersionedHLO_DynamicReshapeOp: VersionedHLO_Op<"dynamic_reshape"> {
  let arguments = (ins VHLO_Tensor:$operand, VHLO_DimensionTensor:$output_shape);
  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_ScatterOp: VersionedHLO_OpWithTraits<"scatter", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    TensorOf<[AnyInteger, Index]>:$scatter_indices,
    Variadic<VHLO_Tensor>:$updates,
    VersionedHLO_ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unique_indices
  );

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs Variadic<VHLO_Tensor>);
}

def VersionedHLO_SelectOp: VersionedHLO_Op<"select"> {
  let arguments = (ins
    VHLO_PredTensor:$pred,
    VHLO_Tensor:$on_true,
    VHLO_Tensor:$on_false
  );

  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_SelectAndScatterOp: VersionedHLO_Op<"select_and_scatter"> {
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

def VersionedHLO_SetDimensionSizeOp: VersionedHLO_Op<"set_dimension_size"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I32Tensor:$size,
    I64Attr:$dimension
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_SortOp : VersionedHLO_Op<"sort"> {
  let arguments = (ins
    Variadic<VHLO_Tensor>:$inputs,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable
  );

  let regions = (region SizedRegion<1>:$comparator);

  let results = (outs Variadic<VHLO_Tensor>);
}

def VersionedHLO_ReverseOp: VersionedHLO_Op<"reverse"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );

  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_PadOp: VersionedHLO_Op<"pad"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high,
    I64ElementsAttr:$interior_padding
  );

  let results = (outs VHLO_Tensor);
}

def VersionedHLO_TraceOp: VersionedHLO_Op<"trace"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    StrAttr:$tag
  );
}

def VersionedHLO_TransposeOp: VersionedHLO_Op<"transpose"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_TriangularSolveOp: VersionedHLO_Op<"triangular_solve"> {
  let arguments = (ins
    VHLO_FpOrComplexTensor:$a,
    VHLO_FpOrComplexTensor:$b,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    VersionedHLO_TransposeAttr:$transpose_a
  );
  let results = (outs VHLO_FpOrComplexTensor);
}

def VersionedHLO_ReduceWindowOp: VersionedHLO_OpWithTraits<"reduce_window", [SameVariadicOperandSize]> {
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

def VersionedHLO_TorchIndexSelectOp : VersionedHLO_Op<"torch_index_select"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$index,
    I64Attr:$dim,
    I64Attr:$batch_dims
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_OptimizationBarrierOp : VersionedHLO_Op<"optimization_barrier"> {
  let arguments = (ins Variadic<VHLO_TensorOrToken>:$operand);
  let results = (outs Variadic<VHLO_TensorOrToken>:$result);
}

//===----------------------------------------------------------------------===//
// VersionedHLO RNG Operators.
//===----------------------------------------------------------------------===//

def VersionedHLO_RngOp : VersionedHLO_Op<"rng"> {
  let arguments = (ins
    0DTensorOf<[VHLO_Pred, VHLO_Int, VHLO_Float]>:$a,
    0DTensorOf<[VHLO_Pred, VHLO_Int, VHLO_Float]>:$b,
    VHLO_DimensionTensor:$shape,
    VersionedHLO_RngDistributionAttr:$rng_distribution
  );
  let results = (outs VHLO_PredIntOrFpTensor:$result);
}

def VersionedHLO_RngBitGeneratorOp : VersionedHLO_Op<"rng_bit_generator"> {
  let arguments = (ins
    VersionedHLO_RngAlgorithmAttr:$rng_algorithm,
    VHLO_IntOrFpTensor:$initial_state
  );

  let results = (outs
    VHLO_IntOrFpTensor:$output_state,
    VHLO_IntOrFpTensor:$output
  );

  
}

// Quantize Ops
def VersionedHLO_UniformQuantizeOp : VersionedHLO_Op<"uniform_quantize"> {
  let arguments = (ins TensorOf<[F32, BF16, VHLO_QuantizedInt]>:$operand);
  let results = (outs VHLO_QuantizedIntTensor:$result);
}

def VersionedHLO_UniformDequantizeOp : VersionedHLO_Op<"uniform_dequantize"> {
  let arguments = (ins VHLO_QuantizedIntTensor:$operand);
  let results = (outs TensorOf<[F32, BF16]>:$result);
}

def VersionedHLO_ReducePrecisionOp : VersionedHLO_Op<"reduce_precision"> {
  let arguments = (ins
    VHLO_FpTensor:$operand,
    I32Attr:$exponent_bits,
    I32Attr:$mantissa_bits
  );
  let results = (outs VHLO_FpTensor:$output);

}

def VersionedHLO_RealDynamicSliceOp: VersionedHLO_Op<"real_dynamic_slice"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_DimensionTensor:$start_indices,
    VHLO_DimensionTensor:$limit_indices,
    VHLO_DimensionTensor:$strides
  );
  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_DynamicPadOp: VersionedHLO_Op<"dynamic_pad"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_Tensor:$padding_value,
    VHLO_DimensionTensor:$edge_padding_low,
    VHLO_DimensionTensor:$edge_padding_high,
    VHLO_DimensionTensor:$interior_padding
  );
  let results = (outs VHLO_Tensor:$result);
}

def VersionedHLO_DynamicGatherOp: VersionedHLO_Op<"dynamic_gather"> {
  let arguments = (ins
    VHLO_Tensor:$operand,
    VHLO_IntTensor:$start_indices,
    VHLO_IntTensor:$slice_sizes,
    VersionedHLO_GatherDimensionNumbers:$dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_DynamicConvOp : VersionedHLO_Op<"dynamic_conv"> {
  let arguments = !con(
    (ins
       VHLO_Tensor:$lhs,
       VHLO_Tensor:$rhs,
       VHLO_Tensor:$d_padding),
    VersionedHLO_ConvolutionAttributes.attributes);
  let results = (outs VHLO_Tensor);
}

def VersionedHLO_ComputeReshapeShapeOp : VersionedHLO_Op<"compute_reshape_shape"> {
  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs 1DTensorOf<[AnyInteger, Index]>:$result);
}

def VersionedHLO_CstrReshapableOp : VersionedHLO_Op<"cstr_reshapable"> {
  let results = (outs Shape_WitnessType:$result);
  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
}

#endif // STABLEHLO_COMPATIBILITY_STABLEHLO_OPS
