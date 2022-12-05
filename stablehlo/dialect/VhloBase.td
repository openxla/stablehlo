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

#ifndef STABLEHLO_DIALECT_VHLO_BASE
#define STABLEHLO_DIALECT_VHLO_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// VHLO Versioning Interfaces
//===----------------------------------------------------------------------===//

def VersionedInterface : OpInterface<"VersionedInterface"> {
  let methods = [
    InterfaceMethod<
      "Returns the minimum version an op is supported in.",
      "mlir::vhlo::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum verison an op is supported in.",
      "mlir::vhlo::Version", "getMaxVersion">,
  ];
}

//===----------------------------------------------------------------------===//
// VHLO Type Definitions.
//===----------------------------------------------------------------------===//

def VHLO_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

// TODO(hinsu): Use signed integers instead of signless integer which is being
// used for legacy reasons.
def VHLO_SInt : SignlessIntOfWidths<[4, 8, 16, 32, 64]>;
def VHLO_UInt : UnsignedIntOfWidths<[4, 8, 16, 32, 64]>;
def VHLO_Int : AnyTypeOf<[VHLO_SInt, VHLO_UInt]>;

def VHLO_Float : AnyTypeOf<[F16, F32, F64, BF16]>;
def VHLO_Float32Or64 : AnyTypeOf<[F32, F64]>;

def VHLO_Complex : Complex<AnyTypeOf<[F32, F64]>>;

//===----------------------------------------------------------------------===//
// Quantized element type definitions.
//===----------------------------------------------------------------------===//

// TODO(b/230381284): Upstream width-specific uniform quantized element types.
class VHLO_UniformQuantizedSignedInt<int width>
  : Type<Or<[
      And<[CPred<"$_self.isa<mlir::quant::UniformQuantizedType>()">,
           CPred<"$_self.cast<mlir::quant::UniformQuantizedType>()" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"$_self.cast<mlir::quant::UniformQuantizedType>()" #
                 ".isSigned()">]>,
      And<[CPred<"$_self.isa<mlir::quant::UniformQuantizedPerAxisType>()">,
           CPred<"$_self.cast<mlir::quant::UniformQuantizedPerAxisType>()" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"$_self.cast<mlir::quant::UniformQuantizedPerAxisType>()" #
                 ".isSigned()">]>]>,
    "QI" # width # " type"> {
  string name = "UniformQuantizedSignedInt";
  int bitwidth = width;
}

class VHLO_UniformQuantizedUnsignedInt<int width>
  : Type<Or<[
      And<[CPred<"$_self.isa<mlir::quant::UniformQuantizedType>()">,
           CPred<"$_self.cast<mlir::quant::UniformQuantizedType>()" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"!$_self.cast<mlir::quant::UniformQuantizedType>()" #
                 ".isSigned()">]>,
      And<[CPred<"$_self.isa<mlir::quant::UniformQuantizedPerAxisType>()">,
           CPred<"$_self.cast<mlir::quant::UniformQuantizedPerAxisType>()" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"!$_self.cast<mlir::quant::UniformQuantizedPerAxisType>()" #
                 ".isSigned()">]>]>,
    "QUI" # width # " type"> {
  string name = "UniformQuantizedUnsignedInt";
  int bitwidth = width;
}

class VHLO_UniformQuantizedSignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, VHLO_UniformQuantizedSignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized signed " #
              "integer">;

class VHLO_UniformQuantizedUnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, VHLO_UniformQuantizedUnsignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized unsigned " #
              "integer">;

// Integer-based uniform quantized types. The definitions can be used to specify
// operand's tensor types.
def VHLO_QuantizedSignedInt : VHLO_UniformQuantizedSignedIntOfWidths<[4, 8, 16, 32]>;
def VHLO_QuantizedUnsignedInt : VHLO_UniformQuantizedUnsignedIntOfWidths<[4, 8, 16, 32]>;
def VHLO_QuantizedInt :
    AnyTypeOf<[VHLO_QuantizedSignedInt, VHLO_QuantizedUnsignedInt]>;

// The broadcasting dimensions correspond to a tuple that describes how a
// smaller rank shape is broadcast into a larger rank shape. For example,
// given a 2x3x4 cuboid and a 3x4 matrix, a broadcasting tuple (1,2) means
// matching the matrix to dimensions 1 and 2 of the cuboid.
defvar BroadcastDimAttr = I64ElementsAttr;

// Token type.
def VHLO_Token : Type<CPred<"$_self.isa<TokenType>()">, "token">;

// Any integer tensor types
def VHLO_IntTensor : TensorOf<[VHLO_Int]>;

// Any integer tensor type with rank 0 (i.e. representing a single integer).
def VHLO_ScalarIntTensor : 0DTensorOf<[VHLO_Int]>;

// Any floating-point tensor types
def VHLO_FpTensor : TensorOf<[VHLO_Float]>;

// 32 or 64 bits floating-point tensor types
def VHLO_Fp32Or64Tensor : TensorOf<[VHLO_Float32Or64]>;

// Any quantized integer tensor types
def VHLO_QuantizedIntTensor : TensorOf<[VHLO_QuantizedInt]>;

def VHLO_PredTensor : TensorOf<[VHLO_Pred]>;

def VHLO_Tensor : TensorOf<[VHLO_Float, VHLO_Pred, VHLO_Int, VHLO_Complex, VHLO_QuantizedInt]>;

def VHLO_ComplexTensor : TensorOf<[VHLO_Complex]>;

def VHLO_Tuple : NestedTupleOf<[VHLO_Tensor, VHLO_Token]>;

def VHLO_TensorOrToken : AnyTypeOf<[VHLO_Tensor, VHLO_Token]>;

def VHLO_TensorOrTokenOrTuple : AnyTypeOf<[VHLO_Tensor, VHLO_Token, VHLO_Tuple]>;

def VHLO_DimensionValue : AnyTypeOf<[Index, VHLO_Int]>;

// Dynamic representation of a shape vector as a tensor.
def VHLO_DimensionTensor : 1DTensorOf<[VHLO_DimensionValue]>;

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
def VHLO_StaticShapeTensor : StaticShapeTensorOf<[
      VHLO_Float, VHLO_Pred, VHLO_Int, VHLO_Complex, VHLO_QuantizedInt]>;

//===----------------------------------------------------------------------===//
// VHLO combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or floating-point tensor types
def VHLO_IntOrFpTensor : TensorOf<[VHLO_Int, VHLO_Float]>;

// Any integer or predicate tensor types
def VHLO_PredOrIntTensor : TensorOf<[VHLO_Pred, VHLO_Int]>;

// Any floating-point or complex tensor types
def VHLO_FpOrComplexTensor : TensorOf<[VHLO_Float, VHLO_Complex]>;

// Any int, floating-point or complex tensor types
def VHLO_IntFpOrComplexTensor : TensorOf<[VHLO_Int, VHLO_Float, VHLO_Complex]>;

// Any pred, int or floating-point tensor types
def VHLO_PredIntOrFpTensor : TensorOf<[VHLO_Pred, VHLO_Int, VHLO_Float]>;

//===----------------------------------------------------------------------===//
// VHLO traits
//===----------------------------------------------------------------------===//

// VHLO intentionally does not include traits since it is only used to represent
// the in-memory structure of the IR at a given version.
//
// This section is included for file parity between VhloBase.td and Base.td

#endif // STABLEHLO_DIALECT_VHLO_BASE
