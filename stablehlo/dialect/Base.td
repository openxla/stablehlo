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

#ifndef STABLEHLO_DIALECT_BASE
#define STABLEHLO_DIALECT_BASE

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Common type definitions.
//===----------------------------------------------------------------------===//

def I32RankedTensor : RankedTensorOf<[I32]>;

def UI32RankedTensor : RankedTensorOf<[UI32]>;

//===----------------------------------------------------------------------===//
// HLO type definitions.
//===----------------------------------------------------------------------===//

def HLO_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

// TODO(hinsu): Use signed integers instead of signless integer which is being
// used for legacy reasons.
def HLO_SInt : SignlessIntOfWidths<[2, 4, 8, 16, 32, 64]>;
def HLO_UInt : UnsignedIntOfWidths<[2, 4, 8, 16, 32, 64]>;
def HLO_Int : AnyTypeOf<[HLO_SInt, HLO_UInt]>;

def HLO_Float : AnyTypeOf<[F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
                           F8E4M3FN, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2,
                           F8E5M2FNUZ, F8E8M0FNU, F16, F32, F64, BF16]>;
def HLO_Float32Or64 : AnyTypeOf<[F32, F64]>;

def HLO_Complex : Complex<AnyTypeOf<[F32, F64]>>;

//===----------------------------------------------------------------------===//
// Quantized element type definitions.
//===----------------------------------------------------------------------===//

def IsValidStablehloQuantizedElementType : CPred<"mlir::hlo::isValidStablehloQuantizedElementType($_self)">;
def IsValidQuantizedDimension : CPred<"mlir::hlo::isValidQuantizedDimension($_self)">;

// TODO(b/230381284): Upstream width-specific uniform quantized element types.
class StableHLO_UniformQuantizedSignedInt<int width>
  : Type<And<[CPred<"isa<mlir::quant::UniformQuantizedType>($_self)">,
           CPred<"cast<mlir::quant::UniformQuantizedType>($_self)" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"cast<mlir::IntegerType>(" #
                   "cast<mlir::quant::UniformQuantizedType>($_self).getStorageType()" #
                 ").isSignless()">, IsValidStablehloQuantizedElementType]>,
    "QI" # width # " type"> {
  string name = "UniformQuantizedSignedInt";
  int bitwidth = width;
}

class StableHLO_UniformQuantizedPerAxisSignedInt<int width>
  : Type<And<[CPred<"isa<mlir::quant::UniformQuantizedPerAxisType>($_self)">,
           CPred<"cast<mlir::quant::UniformQuantizedPerAxisType>($_self)" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"cast<mlir::IntegerType>(" #
                   "cast<mlir::quant::UniformQuantizedPerAxisType>($_self).getStorageType()" #
                 ").isSignless()">, IsValidStablehloQuantizedElementType]>,
    "QI" # width # " type"> {
  string name = "UniformQuantizedPerAxisSignedInt";
  int bitwidth = width;
}

class StableHLO_UniformQuantizedUnsignedInt<int width>
  : Type<And<[CPred<"isa<mlir::quant::UniformQuantizedType>($_self)">,
           CPred<"cast<mlir::quant::UniformQuantizedType>($_self)" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"!cast<mlir::quant::UniformQuantizedType>($_self)" #
                 ".isSigned()">, IsValidStablehloQuantizedElementType]>,
    "QUI" # width # " type"> {
  string name = "UniformQuantizedUnsignedInt";
  int bitwidth = width;
}

class StableHLO_UniformQuantizedPerAxisUnsignedInt<int width>
  : Type<And<[CPred<"isa<mlir::quant::UniformQuantizedPerAxisType>($_self)">,
           CPred<"cast<mlir::quant::UniformQuantizedPerAxisType>($_self)" #
                 ".getStorageTypeIntegralWidth() == " # width>,
           CPred<"!cast<mlir::quant::UniformQuantizedPerAxisType>($_self)" #
                 ".isSigned()">, IsValidStablehloQuantizedElementType]>,
    "QUI" # width # " type"> {
  string name = "UniformQuantizedPerAxisUnsignedInt";
  int bitwidth = width;
}

class StableHLO_UniformQuantizedSignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, StableHLO_UniformQuantizedSignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized signed " #
              "integer">;

class StableHLO_UniformQuantizedUnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, StableHLO_UniformQuantizedUnsignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized unsigned " #
              "integer">;

class StableHLO_UniformQuantizedPerAxisSignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, StableHLO_UniformQuantizedPerAxisSignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized per axis signed " #
              "integer">;

class StableHLO_UniformQuantizedPerAxisUnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, StableHLO_UniformQuantizedPerAxisUnsignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized per axis unsigned " #
              "integer">;

// Integer-based uniform per-tensor quantized types. The definitions can be used to specify
// operand's tensor types.
def HLO_QuantizedSignedInt : StableHLO_UniformQuantizedSignedIntOfWidths<[2, 4, 8, 16, 32]>;
def HLO_QuantizedUnsignedInt : StableHLO_UniformQuantizedUnsignedIntOfWidths<[2, 4, 8, 16, 32]>;
def HLO_QuantizedInt :
    AnyTypeOf<[HLO_QuantizedSignedInt, HLO_QuantizedUnsignedInt]>;

// Integer-based uniform per-axis quantized types. The definitions can be used to specify
// operand's tensor types.
def HLO_PerAxisQuantizedSignedInt : StableHLO_UniformQuantizedPerAxisSignedIntOfWidths<[2, 4, 8, 16, 32]>;
def HLO_PerAxisQuantizedUnsignedInt : StableHLO_UniformQuantizedPerAxisUnsignedIntOfWidths<[2, 4, 8, 16, 32]>;
def HLO_PerAxisQuantizedInt :
    AnyTypeOf<[HLO_PerAxisQuantizedSignedInt, HLO_PerAxisQuantizedUnsignedInt]>;

// Token type.
def HLO_Token : Type<CPred<"isa<TokenType>($_self)">, "token">;

// Any integer tensor types
def HLO_IntTensor : RankedTensorOf<[HLO_Int]>;

// Any integer tensor type with rank 0 (i.e. representing a single integer).
def HLO_ScalarIntTensor : 0DTensorOf<[HLO_Int]>;

// Any floating-point tensor types
def HLO_FpTensor : RankedTensorOf<[HLO_Float]>;

// 32 or 64 bits floating-point tensor types
def HLO_Fp32Or64Tensor : RankedTensorOf<[HLO_Float32Or64]>;

// per-tensor quantized integer tensor types
def HLO_QuantizedIntTensor : RankedTensorOf<[HLO_QuantizedInt]>;

// per-axis quantized integer tensor type
def HLO_PerAxisQuantizedIntTensor : RankedTensorOf<[HLO_PerAxisQuantizedInt],
    [IsValidQuantizedDimension]>;

// per-tensor or per-axis quantized integer tensor type
def HLO_QuantizedIntOrPerAxisQuantizedIntTensor : RankedTensorOf<[HLO_QuantizedInt, HLO_PerAxisQuantizedInt],
    [IsValidQuantizedDimension]>;

def HLO_PredTensor : RankedTensorOf<[HLO_Pred]>;

def HLO_Tensor : RankedTensorOf<[HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt]>;

def HLO_ScalarTensor: 0DTensorOf<[HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt]>;

def HLO_NonQuantizedTensor : RankedTensorOf<[HLO_Float, HLO_Pred, HLO_Int, HLO_Complex]>;

def HLO_TensorOrPerAxisQuantizedTensor : RankedTensorOf<[HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt, HLO_PerAxisQuantizedInt],
    [IsValidQuantizedDimension]>;

def HLO_ComplexTensor : RankedTensorOf<[HLO_Complex]>;

def HLO_Tuple : NestedTupleOf<[HLO_Tensor, HLO_PerAxisQuantizedIntTensor, HLO_Token]>;

def HLO_TensorOrToken : AnyTypeOf<[HLO_Tensor, HLO_Token]>;

def HLO_TensorOrPerAxisQuantizedTensorOrToken : AnyTypeOf<[HLO_Tensor, HLO_PerAxisQuantizedIntTensor, HLO_Token]>;

def HLO_TensorOrTokenOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Token, HLO_Tuple]>;

def HLO_TensorOrPerAxisQuantizedTensorOrTokenOrTuple : AnyTypeOf<[HLO_Tensor, HLO_PerAxisQuantizedIntTensor, HLO_Token, HLO_Tuple]>;

def HLO_DimensionValue : AnyTypeOf<[Index, HLO_Int]>;

// Dynamic representation of a shape vector as a tensor.
def HLO_DimensionTensor : 1DTensorOf<[HLO_DimensionValue]>;

//===----------------------------------------------------------------------===//
// Exceptions for unranked dynamism. These should not be used with StableHLO,
// but may be used with CHLO for now.
// TODO(b/326463552): Remove these when CHLO no longer needs unranked dynamism.
//===----------------------------------------------------------------------===//

def HLO_AnyTensor : TensorOf<[HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt, HLO_PerAxisQuantizedInt],
    [IsValidQuantizedDimension]>;

def HLO_AnyPredTensor : TensorOf<[HLO_Pred]>;

def HLO_AnyFpTensor : TensorOf<[HLO_Float]>;

def HLO_AnyComplexTensor : TensorOf<[HLO_Complex]>;

def HLO_AnyFpOrComplexTensor : TensorOf<[HLO_Float, HLO_Complex]>;

def HLO_AnyPredOrIntTensor : TensorOf<[HLO_Pred, HLO_Int]>;

def HLO_AnyTuple : NestedTupleOf<[HLO_AnyTensor, HLO_Token]>;

def HLO_CustomCallValue : AnyTypeOf<[HLO_AnyTensor, HLO_Token, HLO_AnyTuple]>;

//===----------------------------------------------------------------------===//
// HLO combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or floating-point tensor types
def HLO_IntOrFpTensor : RankedTensorOf<[HLO_Int, HLO_Float]>;

// Any integer or predicate tensor types
def HLO_PredOrIntTensor : RankedTensorOf<[HLO_Pred, HLO_Int]>;

// Any floating-point or complex tensor types
def HLO_FpOrComplexTensor : RankedTensorOf<[HLO_Float, HLO_Complex]>;

// Any floating-point or quantized tensor types
def HLO_FpOrQuantizedIntTensor : RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>;

// Any floating-point, complex or quantized tensor types
def HLO_FpComplexOrQuantizedIntTensor : RankedTensorOf<[HLO_Float, HLO_Complex, HLO_QuantizedInt]>;

// Any int, floating-point, complex or quantized tensor types
def HLO_IntFpOrComplexOrQuantizedIntTensor : RankedTensorOf<[HLO_Int, HLO_Float, HLO_Complex, HLO_QuantizedInt]>;

// Any pred, int or floating-point tensor types
def HLO_PredIntOrFpTensor : RankedTensorOf<[HLO_Pred, HLO_Int, HLO_Float]>;

// Any pred, int or floating-point ranked tensor types
// TODO: delete this type once ranked tensors are the default
def HLO_PredIntOrFpRankedTensor : RankedTensorOf<[HLO_Pred, HLO_Int, HLO_Float]>;

// Any pred, int, floating-point or quantized tensor types
def HLO_PredIntFpOrQuantizedTensor : RankedTensorOf<[HLO_Pred, HLO_Int, HLO_Float, HLO_QuantizedInt]>;

//===----------------------------------------------------------------------===//
// HLO static shape type definitions.
//===----------------------------------------------------------------------===//

// Static representation of a shape vector as a tensor.
def HLO_StaticDimensionTensor : RankedTensorOf<[HLO_DimensionValue], [HasStaticShapePred, HasAnyRankOfPred<[1]>], "statically shaped 1-dimensional tensor">;

// Static representation of a 1D tensor of int.
def HLO_Static1DIntTensor : RankedTensorOf<[HLO_Int], [HasStaticShapePred, HasAnyRankOfPred<[1]>], "statically shaped 1-dimensional integer tensor">;

// Static representation of a 2D tensor of int.
def HLO_Static2DIntTensor : RankedTensorOf<[HLO_Int], [HasStaticShapePred, HasAnyRankOfPred<[2]>], "statically shaped 2-dimensional integer tensor">;

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
def HLO_StaticShapeTensor : StaticShapeTensorOf<[
      HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt]>;

def HLO_StaticShapeTensorOrPerAxisQuantizedTensor : RankedTensorOf<[HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt, HLO_PerAxisQuantizedInt],
    [IsValidQuantizedDimension, HasStaticShapePred], "statically shaped tensor">;

def HLO_StaticShapeTensorOrPerAxisQuantizedTensorOrToken : AnyTypeOf<[HLO_StaticShapeTensor, HLO_StaticShapeTensorOrPerAxisQuantizedTensor, HLO_Token]>;

def HLO_StaticShapeIntOrFpTensor : StaticShapeTensorOf<[HLO_Int, HLO_Float]>;

def HLO_StaticShapeIntFpOrComplexTensor :
  StaticShapeTensorOf<[HLO_Int, HLO_Float, HLO_Complex]>;

def HLO_StaticShapeIntFpComplexOrQuantizedTensor :
  StaticShapeTensorOf<[HLO_Int, HLO_Float, HLO_Complex, HLO_QuantizedInt]>;

//===----------------------------------------------------------------------===//
// HLO traits
//===----------------------------------------------------------------------===//

class HLO_NativeOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::hlo::OpTrait";
}

// An operation that is essentially element-wise but may implement broadcasting
// semantics.
def HLO_BroadcastingElementwise : HLO_NativeOpTrait<"BroadcastingElementwise">;

// This class adds property that the operation is commutative.
// Upstream IsCommutative has default folders, and StableHLO aims to have no
// default folders or canonicalizations.
def HLO_Commutative : HLO_NativeOpTrait<"IsCommutative">;

// Op has pairwise operand and result type matching: the number of operands
// must be equal to the number of results and the type of ith operand must
// match the type of ith result.
// TODO(b/195086460) Promote this to be an mlir trait and remove it here.
def HLO_PairwiseSameOperandAndResultType :
  HLO_NativeOpTrait<"PairwiseSameOperandAndResultType">;

// Op has pairwise operand and result element type matching: the number of operands
// must be equal to the number of results and the element type of ith operand must
// match the element type of ith result.
def HLO_PairwiseSameOperandAndResultElementType :
  HLO_NativeOpTrait<"PairwiseSameOperandAndResultElementType">;

// Op has operand and result types compatible with each other according to
// the rules implemented in isCompatibleForHloTypeInference, which account for
// special properties dynamism, quantization and sparsity.
def HLO_CompatibleOperandsAndResultType : TraitList<
  // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
  // support quantization or sparsity.
  [
    InferTypeOpInterface,
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
    HLO_NativeOpTrait<"CompatibleOperandsAndResultType">
  ]>;

def HLO_CompatibleOperandsAndResultElementType :
  HLO_NativeOpTrait<"CompatibleOperandsAndResultElementType">;

def HLO_CompatibleOperandsElementType :
  HLO_NativeOpTrait<"CompatibleOperandsElementType">;

def HLO_BoundedAttrInterface : AttrInterface<"BoundedAttrInterface"> {
  let cppNamespace = "::mlir::hlo";

  let description = [{
    This interface is used for attributes that carry bounds for dimension sizes
    of an accompanying shaped type, e.g. when the attribute represents a
    RankedTensorType::getEncoding.
    The number of bounds is expected to be the same as the number of dimensions
    in the accompanying shaped type.
    For a static dimension, the corresponding bound is ShapedType::kDynamic.
    For a dynamic dimension, the corresponding bound is either known and is
    a non-negative number or unknown and is ShapedType::kDynamic.
  }];

  let methods = [InterfaceMethod<
    "Get the attribute's bounds",
    "::llvm::ArrayRef<int64_t>", "getBounds"
  >];
}

def HLO_SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait
  : HLO_NativeOpTrait<"SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait">;

// This trait can be used with ops where the result has the same rank as the
// input and each dimension of the input maps to the same dimension in the
// result. In that case, if a dim is static in the output but dynamic in the
// input, the dimension could differ at runtime, leading to undefined behavior.
// If the output dimension is dynamic, there is no expectation, so there
// cannot be a mismatch. If the input dimension is static, the output dimension
// can be inferred from it, so there cannot be a mismatch either.
def HLO_SpeculatableIfStaticDimInOutputIsStaticInInput : TraitList<[
    ConditionallySpeculatable, HLO_SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait]>;

def HLO_RecursivelySpeculatableIfStaticDimInOutputIsStaticInInputImplTrait
  : HLO_NativeOpTrait<"RecursivelySpeculatableIfStaticDimInOutputIsStaticInInputImplTrait">;

// This trait is the same as HLO_SpeculatableIfStaticDimInOutputIsStaticInInput,
// but for ops that have regions. If all static dimensions in the output are
// static in the input, such an op is RecursivelySpeculatable (the ops in its
// regions have to be checked for speculatability).
def HLO_RecursivelySpeculatableIfStaticDimInOutputIsStaticInInput : TraitList<[
    ConditionallySpeculatable, HLO_RecursivelySpeculatableIfStaticDimInOutputIsStaticInInputImplTrait]>;

def HLO_SpeculatableIfAllInputsStaticImplTrait
  : HLO_NativeOpTrait<"SpeculatableIfAllInputsStaticImplTrait">;

// This trait is appropriate to use for ops where the op is only speculatable
// if all the inputs are static. This can happen if e.g. the inputs are expected
// to all have the same shape. If all the inputs are static, the verifier can
// validate statically that all the static dimensions are the same. However,
// if there is any dynamic dimension, it could differ from the shape of another
// input at runtime, leading to undefined behavior.
def HLO_SpeculatableIfAllInputsStatic : TraitList<[
    ConditionallySpeculatable, HLO_SpeculatableIfAllInputsStaticImplTrait]>;

def HLO_RecursivelySpeculatableIfAllInputsStaticImplTrait
  : HLO_NativeOpTrait<"RecursivelySpeculatableIfAllInputsStaticImplTrait">;

// This trait is the same as HLO_SpeculatableIfAllInputsStatic, but for ops that
// have regions. If all the inputs are static, such an op is
// RecursivelySpeculatable (the ops in its regions have to be checked for
// speculatability).
def HLO_RecursivelySpeculatableIfAllInputsStatic : TraitList<[
    ConditionallySpeculatable, HLO_RecursivelySpeculatableIfAllInputsStaticImplTrait]>;

def HLO_SpeculatableIfAllInputsStaticAndShapeConstantImplTrait
  : HLO_NativeOpTrait<"SpeculatableIfAllInputsStaticAndShapeConstantImplTrait">;

// This trait is the same as HLO_SpeculatableIfAllInputsStatic, but for ops that
// take a shape as their last operand. Such ops are speculatable if all inputs
// are static and the shape is constant.
def HLO_SpeculatableIfAllInputsStaticAndShapeConstant : TraitList<[
    ConditionallySpeculatable, HLO_SpeculatableIfAllInputsStaticAndShapeConstantImplTrait]>;

#endif // STABLEHLO_DIALECT_BASE
