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

#ifndef STABLEHLO_COMPATIBILITY_STABLEHLOV1_OPS
#define STABLEHLO_COMPATIBILITY_STABLEHLOV1_OPS

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "stablehlo/dialect/Base.td"

//===----------------------------------------------------------------------===//
// HLOv1 type definitions.
//===----------------------------------------------------------------------===//

def HLOv1_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

// TODO(hinsu): Use signed integers instead of signless integer which is being
// used for legacy reasons.
def HLOv1_SInt : SignlessIntOfWidths<[4, 8, 16, 32, 64]>;
def HLOv1_UInt : UnsignedIntOfWidths<[4, 8, 16, 32, 64]>;
def HLOv1_Int : AnyTypeOf<[HLO_SInt, HLO_UInt]>;

def HLOv1_Float : AnyTypeOf<[F16, F32, F64, BF16]>;
def HLOv1_Float32Or64 : AnyTypeOf<[F32, F64]>;

def HLOv1_Complex : Complex<AnyTypeOf<[F32, F64]>>;

def StableHLOv1_Dialect : Dialect {
  let name = "stablehlo_v1";
  let cppNamespace = "::mlir::versioned_stablehlo";

  let description = [{
    A shim opset of versioned StableHLO ops for versions 0.x.x and 1.x.x.
  }];
}

class StableHLOv1_Op<string mnemonic> :
    Op<StableHLOv1_Dialect, mnemonic, []>;

//include "stablehlo/compatibility/VersionedStablehloEnums.td"
//include "stablehlo/compatibility/VersionedStablehloAttrs.td"

//===----------------------------------------------------------------------===//
// StableHLO v1 unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class StableHLOv1_UnaryElementwiseOp<string mnemonic, Type OperandType, Type ResultType = OperandType> 
        : StableHLOv1_Op<mnemonic> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
}

// Abs supports complex to real, so element type is not guaranteed to match.
def StableHLOv1_AbsOp: StableHLOv1_UnaryElementwiseOp<"abs_v0", TensorOf<[HLOv1_SInt, HLOv1_Float, HLOv1_Complex]>>;
/*
def StableHLOv1_CbrtOp: StableHLOv1_UnaryElementwiseOp<"cbrt_v0">;
def StableHLOv1_CeilOp: StableHLOv1_UnaryElementwiseOp<"ceil_v0">; 
def StableHLOv1_ConvertOp : StableHLOv1_UnaryElementwiseOp<"convert_v0">;
def StableHLOv1_ClzOp: StableHLOv1_UnaryElementwiseOp<"count_leading_zeros_v0">;
def StableHLOv1_CosineOp: StableHLOv1_UnaryElementwiseOp<"cosine_v0">;
def StableHLOv1_ExpOp: StableHLOv1_UnaryElementwiseOp<"exponential_v0">;
def StableHLOv1_Expm1Op: StableHLOv1_UnaryElementwiseOp<"exponential_minus_one_v0">;
def StableHLOv1_FloorOp: StableHLOv1_UnaryElementwiseOp<"floor_v0">;
def StableHLOv1_ImagOp: StableHLOv1_UnaryElementwiseOp<"imag_v0">;
def StableHLOv1_IsFiniteOp: StableHLOv1_UnaryElementwiseOp<"is_finite_v0"> {
  let arguments = (ins HLO_FpTensor:$x);
  let results = (outs HLO_PredTensor:$y);
}
def StableHLOv1_LogOp: StableHLOv1_UnaryElementwiseOp<"log_v0">;
def StableHLOv1_Log1pOp: StableHLOv1_UnaryElementwiseOp<"log_plus_one_v0">;
def StableHLOv1_LogisticOp: StableHLOv1_UnaryElementwiseOp<"logistic_v0">;
def StableHLOv1_NotOp: StableHLOv1_UnaryElementwiseOp<"not_v0">;
def StableHLOv1_NegOp: StableHLOv1_UnaryElementwiseOp<"negate_v0">;
def StableHLOv1_PopulationCountOp: StableHLOv1_UnaryElementwiseOp<"popcnt_v0">;
def StableHLOv1_RealOp: StableHLOv1_UnaryElementwiseOp<"real_v0">;
def StableHLOv1_RoundOp: StableHLOv1_UnaryElementwiseOp<"round_nearest_afz_v0">;
def StableHLOv1_RoundNearestEvenOp: StableHLOv1_UnaryElementwiseOp<"round_nearest_even">;
def StableHLOv1_RsqrtOp: StableHLOv1_UnaryElementwiseOp<"rsqrt">;
def StableHLOv1_SignOp: StableHLOv1_UnaryElementwiseOp<"sign">;
def StableHLOv1_SineOp: StableHLOv1_UnaryElementwiseOp<"sine">;
def StableHLOv1_SqrtOp: StableHLOv1_UnaryElementwiseOp<"sqrt">;
def StableHLOv1_TanhOp: StableHLOv1_UnaryElementwiseOp<"tanh">;
//===----------------------------------------------------------------------===//
// StableHLOv1 binary elementwise op definitions.
//===----------------------------------------------------------------------===//

class StableHLOv1_BinaryElementwiseOp<string mnemonic> : StableHLOv1_Op<mnemonic> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs
  );
  let results = (outs HLO_Tensor:$result);
}

def StableHLOv1_AddOp : StableHLOv1_BinaryElementwiseOp<"add">;
def StableHLOv1_Atan2Op : StableHLOv1_BinaryElementwiseOp<"atan2">;
def StableHLOv1_ComplexOp: StableHLOv1_BinaryElementwiseOp<"complex"> {
  let arguments = (ins HLO_Fp32Or64Tensor:$lhs, HLO_Fp32Or64Tensor:$rhs);
  let results = (outs HLO_ComplexTensor:$result);
}
def StableHLOv1_DivOp : StableHLOv1_BinaryElementwiseOp<"divide">;
def StableHLOv1_MaxOp : StableHLOv1_BinaryElementwiseOp<"maximum">;
def StableHLOv1_MinOp : StableHLOv1_BinaryElementwiseOp<"minimum">;
def StableHLOv1_MulOp : StableHLOv1_BinaryElementwiseOp<"multiply">;
def StableHLOv1_PowOp : StableHLOv1_BinaryElementwiseOp<"power">;
def StableHLOv1_RemOp : StableHLOv1_BinaryElementwiseOp<"remainder">;
def StableHLOv1_ShiftLeftOp : StableHLOv1_BinaryElementwiseOp<"shift_left">;
def StableHLOv1_ShiftRightArithmeticOp : StableHLOv1_BinaryElementwiseOp<"shift_right_arithmetic">;
def StableHLOv1_ShiftRightLogicalOp : StableHLOv1_BinaryElementwiseOp<"shift_right_logical">;
def StableHLOv1_SubtractOp : StableHLOv1_BinaryElementwiseOp<"subtract">;


//===----------------------------------------------------------------------===//
// StableHLO v1 binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class StableHLOv1_BinaryBiwiseOrLogicalElementwiseOp<string mnemonic> :
        StableHLOv1_BinaryElementwiseOp<mnemonic> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs
  );
}
def StableHLOv1_AndOp: StableHLOv1_BinaryBiwiseOrLogicalElementwiseOp<"and">;
def StableHLOv1_OrOp: StableHLOv1_BinaryBiwiseOrLogicalElementwiseOp<"or">;
def StableHLOv1_XorOp : StableHLOv1_BinaryBiwiseOrLogicalElementwiseOp<"xor">; 
*/

#endif // STABLEHLO_COMPATIBILITY_STABLEHLO_OPS
