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

#ifndef STABLEHLO_DIALECT_STABLEHLO_OPS
#define STABLEHLO_DIALECT_STABLEHLO_OPS

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "stablehlo/dialect/Base.td"

def StableHLO_Dialect : Dialect {
  let name = "stablehlo";
  let cppNamespace = "::mlir::stablehlo";

  let description = [{
    StableHLO is an operation set that expresses ML computations. It has been
    originally bootstrapped from the MHLO dialect and enhances it with additional
    functionality, including serialization and versioning, to be used as
    a portability layer between ML frameworks and ML compilers.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

class StableHLO_Op<string mnemonic, list<Trait> traits> :
    Op<StableHLO_Dialect, mnemonic, traits> {
  let extraClassDeclaration = [{
    // Relax the strict default implementation with one that allows
    // for StableHLO-specific differences.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

include "stablehlo/dialect/StablehloEnums.td"
include "stablehlo/dialect/StablehloAttrs.td"

class StableHLO_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    StableHLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO nullary op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_ConstantOp : StableHLO_Op<"constant",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

    Example:
    ```mlir
    %output = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    ```
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    HLO_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Attribute":$value)>];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def StableHLO_IotaOp : StableHLO_Op<"iota", [Pure]> {
  let summary = "Iota operation";
  let description = [{
    Fills an `output` tensor with values in increasing order starting from zero
    along the `iota_dimension` dimension.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

    Example:
    ```mlir
    %output = stablehlo.iota dim = 0 : tensor<4x5xi32>
    ```
  }];
  let arguments = (ins I64Attr:$iota_dimension);

  let results = (outs HLO_StaticShapeIntFpOrComplexTensor:$output);

  let hasVerifier = 1;

  let assemblyFormat = "`dim` `=` $iota_dimension attr-dict `:` type($output)";
}

def StableHLO_DynamicIotaOp: StableHLO_ShapedInterfaceOp<"dynamic_iota", [Pure]> {
  let summary = "DynamicIota operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as IotaOp except that the
    result shape is specified dynamically via `output_shape`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

    Example:
    ```mlir
    %0 = stablehlo.dynamic_iota %arg0, dim = 0 : (tensor<1xindex>) -> tensor<4xi32>
    ```
  }];

  let arguments = (ins HLO_DimensionTensor:$output_shape, I64Attr:$iota_dimension);
  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $output_shape `,` `dim` `=` $iota_dimension attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_CreateTokenOp : StableHLO_Op<"create_token", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "CreateToken operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as AfterAllOp with 0 inputs:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

    Example:
    ```mlir
    %output = stablehlo.create_token : !stablehlo.token
    ```
  }];

  let results = (outs HLO_Token:$output);

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// StableHLO unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class StableHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType, Type ResultType = OperandType> : StableHLO_Op<mnemonic, traits # [Elementwise,
    InferShapedTypeOpInterface, SameOperandsAndResultShape]> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                operands.front(),
                                                &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

// Abs supports complex to real, so element type is not guaranteed to match.
def StableHLO_AbsOp: StableHLO_UnaryElementwiseOp<"abs",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
     TensorOf<[HLO_SInt, HLO_Float, HLO_Complex] /* abs_i1 */>,
     TensorOf<[HLO_SInt, HLO_Float]>> {
  let summary = "Abs operation";
  let description = [{
    Performs element-wise abs operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs

    Example:
    ```mlir
    %result = stablehlo.abs %operand : tensor<3xi32>
    ```
  }];
}

def StableHLO_CbrtOp: StableHLO_UnaryElementwiseOp<"cbrt",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Cbrt operation";
  let description = [{
    Performs element-wise cubic root operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cbrt

    Example:
    ```mlir
    %result = stablehlo.cbrt %operand : tensor<4xf32>
    ```
  }];
}

def StableHLO_CeilOp: StableHLO_UnaryElementwiseOp<"ceil",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpTensor> {
  let summary = "Ceil operation";
  let description = [{
    Performs element-wise ceil of `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#ceil

    Example:
    ```mlir
    %result = stablehlo.ceil %operand : tensor<5xf32>
    ```
  }];
}

def StableHLO_ConvertOp : StableHLO_UnaryElementwiseOp<"convert",
    [Pure, SameOperandsAndResultShape], HLO_Tensor> {
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = stablehlo.convert %operand : (tensor<3xi32>) -> tensor<3xcomplex<f32>>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$operand, "Type":$result_element_ty)>];
}

def StableHLO_ClzOp: StableHLO_UnaryElementwiseOp<"count_leading_zeros",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_IntTensor> {
  let summary = "Clz operation";
  let description = [{
    Performs element-wise count of the number of leading zero bits in the
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros

    Example:
    ```mlir
    %result = stablehlo.count_leading_zeros %operand : tensor<2x2xi8>
    ```
  }];
}

def StableHLO_CosineOp: StableHLO_UnaryElementwiseOp<"cosine",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Cosine operation";
  let description = [{
    Performs element-wise cosine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cosine

    Example:
    ```mlir
    %result = stablehlo.cosine %operand : tensor<2xf32>
    ```
  }];
}

def StableHLO_ExpOp: StableHLO_UnaryElementwiseOp<"exponential",
    [Pure, HLO_CompatibleOperandsAndResultType /*exponential_c1*/],
     HLO_FpOrComplexTensor /*exponential_i1*/> {
  let summary = "Exp operation";
  let description = [{
    Performs element-wise exponential operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential

    Example:
    ```mlir
    %result = stablehlo.exponential %operand : tensor<2x2xf64>
    ```
  }];
}

def StableHLO_Expm1Op: StableHLO_UnaryElementwiseOp<"exponential_minus_one",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Expm1 operation";
  let description = [{
    Performs element-wise exponential minus one operation on `operand` tensor
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one

    Example:
    ```mlir
    %result = stablehlo.exponential_minus_one %operand : tensor<2xf32>
    ```
  }];
}

def StableHLO_FloorOp: StableHLO_UnaryElementwiseOp<"floor",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpTensor> {
  let summary = "Floor operation";
  let description = [{
    Performs element-wise floor of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#floor

    Example:
    ```mlir
    %result = stablehlo.floor %operand : tensor<2xf32>
    ```
  }];
}

def StableHLO_ImagOp: StableHLO_UnaryElementwiseOp<"imag",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    HLO_FpOrComplexTensor /*imag_i1*/, HLO_FpTensor> {/*imag_c1*/
  let summary = "Imag operation";
  let description = [{
    Extracts the imaginary part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#imag

    Example:
    ```mlir
    %result = stablehlo.imag %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    ```
  }];
}

def StableHLO_IsFiniteOp: StableHLO_UnaryElementwiseOp<"is_finite", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>], HLO_Tensor> {
  let summary = "IsFinite operation";
  let description = [{
    Performs element-wise check whether the value in `x` is finite (i.e. is
    neither +Inf, -Inf, nor NaN) and produces a `y` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#is_finite

    Example:
    ```mlir
    %y = stablehlo.is_finite %x : (tensor<7xf32>) -> tensor<7xi1>
    ```
  }];
  let arguments = (ins HLO_FpTensor:$x);
  let results = (outs HLO_PredTensor:$y);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_LogOp: StableHLO_UnaryElementwiseOp<"log",
    [Pure, HLO_CompatibleOperandsAndResultType /*log_c1*/],
     HLO_FpOrComplexTensor /*log_i1*/> {
  let summary = "Log operation";
  let description = [{
    Performs element-wise logarithm operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log

    Example:
    ```mlir
    %result = stablehlo.log %operand : tensor<2x2xf64>
    ```
  }];
}

def StableHLO_Log1pOp: StableHLO_UnaryElementwiseOp<"log_plus_one",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Log1p operation";
  let description = [{
    Performs element-wise logarithm plus one operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one

    Example:
    ```mlir
    %result = stablehlo.log_plus_one %operand : tensor<6xf32>
    ```
  }];
}

def StableHLO_LogisticOp: StableHLO_UnaryElementwiseOp<"logistic",
    [Pure, HLO_CompatibleOperandsAndResultType /*logistic_c1*/],
    HLO_FpOrComplexTensor /*logistic_i1*/> { /*logistic_c1*/
  let summary = "Logistic operation";
  let description = [{
    Performs element-wise logistic operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#logistic

    Example:
    ```mlir
    %result = stablehlo.logistic %operand : tensor<2x2xf64>
    ```
  }];
}

def StableHLO_NotOp: StableHLO_UnaryElementwiseOp<"not",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_PredOrIntTensor> {
  let summary = "Not operation";
  let description = [{
    Performs element-wise NOT of tensor `operand` of type integer and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not

    Example:
    ```mlir
    %result = stablehlo.not %operand : tensor<5x3x1xi1>
    ```
  }];
}

def StableHLO_NegOp: StableHLO_UnaryElementwiseOp<"negate",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_IntFpOrComplexTensor> {
  let summary = "Neg operation";
  let description = [{
    Performs element-wise negation of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

    Example:
    ```mlir
    %result = stablehlo.negate %operand : tensor<2x3xi32>
    ```
  }];
}

def StableHLO_PopulationCountOp: StableHLO_UnaryElementwiseOp<"popcnt",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_IntTensor> {
  let summary = "PopulationCount operation";
  let description = [{
    Performs element-wise count of the number of bits set in the `operand`
    tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt

    Example:
    ```mlir
    %result = stablehlo.popcnt %operand : tensor<4xi8>
    ```
  }];
}

def StableHLO_RealOp: StableHLO_UnaryElementwiseOp<"real",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    HLO_FpOrComplexTensor /*real_i1*/, HLO_FpTensor> {/*real_c1*/
  let summary = "Real operation";
  let description = [{
    Extracts the real part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#real

    Example:
    ```mlir
    %result = stablehlo.real %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    ```
  }];
}

def StableHLO_RoundOp: StableHLO_UnaryElementwiseOp<"round_nearest_afz",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpTensor> {
  let summary = "Round operation";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties
    away from zero, on the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_afz

    Example:
    ```mlir
    %result = stablehlo.round_nearest_afz %operand : tensor<5xf32>
    ```
  }];
}

def StableHLO_RoundNearestEvenOp: StableHLO_UnaryElementwiseOp<"round_nearest_even",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpTensor> {
  let summary = "RoundNearestEven operation";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties
    towards the even integer, on the `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_even

    Example:
    ```mlir
    %result = stablehlo.round_nearest_even %operand : tensor<5xf32>
    ```
  }];
}

def StableHLO_RsqrtOp: StableHLO_UnaryElementwiseOp<"rsqrt", [Pure,
    HLO_CompatibleOperandsAndResultType /* rsqrt_c1 */],
    HLO_FpOrComplexTensor /* rsqrt_i1 */> {
  let summary = "Rsqrt operation";
  let description = [{
    Performs element-wise reciprocal square root operation on `operand` tensor
    and produces a `result` tensor, implementing the `rSqrt` operation from the
    IEEE-754 specification.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rsqrt

    Example:
    ```mlir
    %result = stablehlo.rsqrt %operand : tensor<2x2xf32>
    ```
  }];
}

def StableHLO_SignOp: StableHLO_UnaryElementwiseOp<"sign",
    [Pure, HLO_CompatibleOperandsAndResultType],
    TensorOf<[HLO_SInt, HLO_Float, HLO_Complex]>> {
  let summary = "Sign operation";
  let description = [{
    Returns the sign of the `operand` element-wise and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    Example:
    ```mlir
    %result = stablehlo.sign %operand : tensor<7xf32>
    ```
  }];
}

def StableHLO_SineOp: StableHLO_UnaryElementwiseOp<"sine",
    [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Sine operation";
  let description = [{
    Performs element-wise sine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sine

    Example:
    ```mlir
    %result = stablehlo.sine %operand : tensor<2xf32>
    ```
  }];
}

def StableHLO_SqrtOp: StableHLO_UnaryElementwiseOp<"sqrt", [Pure,
    HLO_CompatibleOperandsAndResultType /* sqrt_c1 */],
    HLO_FpOrComplexTensor /* sqrt_i1 */> {
  let summary = "Sqrt operation";
  let description = [{
    Performs element-wise square root operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sqrt

    Example:
    ```mlir
    %result = stablehlo.sqrt %operand : tensor<2x2xf32>
    ```
  }];
}

def StableHLO_TanhOp: StableHLO_UnaryElementwiseOp<"tanh",
    [Pure, HLO_CompatibleOperandsAndResultType],
    HLO_FpOrComplexTensor> {
  let summary = "Tanh operation";
  let description = [{
    Performs element-wise hyperbolic tangent operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tanh

    Example:
    ```mlir
    %result = stablehlo.tanh %operand : tensor<2xf32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class StableHLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType = HLO_Tensor, Type ResultType = OperandType> :
    StableHLO_Op<mnemonic, traits # [InferShapedTypeOpInterface,
    SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
    OperandType:$lhs,
    OperandType:$rhs
  );

  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                 operands.front(),
                                                 &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let results = (outs ResultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def StableHLO_AddOp : StableHLO_BinaryElementwiseOp<"add",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Add operation";
  let description = [{
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add

    Example:
    ```mlir
    %result = stablehlo.add %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
}

def StableHLO_Atan2Op : StableHLO_BinaryElementwiseOp<"atan2",
      [Pure, HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Atan2 operation";
  let description = [{
    Performs element-wise atan2 operation on `lhs` and `rhs` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#atan2

    Example:
    ```mlir
    %result = stablehlo.atan2 %lhs, %rhs : tensor<3xf32>
    ```
  }];
}

def StableHLO_ComplexOp: StableHLO_BinaryElementwiseOp<"complex", [Pure,
    SameOperandsElementType, SameOperandsAndResultShape,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Complex operation";
  let description = [{
    Performs element-wise conversion to a complex value from a pair of real and
    imaginary values, `lhs` and `rhs`, and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#complex

    Example:
    ```mlir
    %result = stablehlo.complex %lhs, %rhs : tensor<2xcomplex<f32>>
    ```
  }];
  let arguments = (ins HLO_Fp32Or64Tensor:$lhs, HLO_Fp32Or64Tensor:$rhs);
  let results = (outs HLO_ComplexTensor:$result);

  let assemblyFormat = [{
    operands attr-dict
      `:` custom<ComplexOpType>(type($lhs), type($rhs), type($result))
  }];
}

def StableHLO_DivOp : StableHLO_BinaryElementwiseOp<"divide", [Pure,
    HLO_CompatibleOperandsAndResultType /* div_c1 */],
    HLO_IntFpOrComplexTensor /* div_i1, div_i2 */> {
  let summary = "Div operation";
  let description = [{
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

    Example:
    ```mlir
    %result = stablehlo.divide %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def StableHLO_MaxOp : StableHLO_BinaryElementwiseOp<"maximum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Max operation";
  let description = [{
    Performs element-wise max operation on tensors `lhs` and `rhs` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum

    Example:
    ```mlir
    %result = stablehlo.maximum %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def StableHLO_MinOp : StableHLO_BinaryElementwiseOp<"minimum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Min operation";
  let description = [{
    Performs element-wise min operation on tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum

    Example:
    ```mlir
    %result = stablehlo.minimum %lhs, %rhs : tensor<4xf32>
    ```
  }];
}

def StableHLO_MulOp : StableHLO_BinaryElementwiseOp<"multiply",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Mul operation";
  let description = [{
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply

    Example:
    ```mlir
    %result = stablehlo.multiply %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

def StableHLO_PowOp : StableHLO_BinaryElementwiseOp<"power",
      [Pure, HLO_CompatibleOperandsAndResultType /* pow_c1 */],
      HLO_IntFpOrComplexTensor /* pow_i1, pow_i2 */> {
  let summary = "Power operation";
  let description = [{
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

    Example:
    ```mlir
    %result = stablehlo.power %lhs, %rhs : tensor<6xf64>
    ```
  }];
}

def StableHLO_RemOp : StableHLO_BinaryElementwiseOp<"remainder",
      [Pure, HLO_CompatibleOperandsAndResultType /*remainder_c1*/],
       HLO_IntFpOrComplexTensor /*remainder_i1, remainder_i2*/> {
  let summary = "Rem operation";
  let description = [{
    Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder

    Example:
    ```mlir
    %result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
    ```
  }];
}

def StableHLO_ShiftLeftOp : StableHLO_BinaryElementwiseOp<"shift_left",
      [Pure, HLO_CompatibleOperandsAndResultType], HLO_IntTensor> {
  let summary = "ShiftLeft operation";
  let description = [{
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs`
    number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left

    Example:
    ```mlir
    %result = stablehlo.shift_left %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def StableHLO_ShiftRightArithmeticOp : StableHLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [Pure, HLO_CompatibleOperandsAndResultType], HLO_IntTensor> {
  let summary = "ShiftRightArithmetic operation";
  let description = [{
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor
    by `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic

    Example:
    ```mlir
    %result = stablehlo.shift_right_arithmetic %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def StableHLO_ShiftRightLogicalOp : StableHLO_BinaryElementwiseOp<"shift_right_logical",
      [Pure, HLO_CompatibleOperandsAndResultType], HLO_IntTensor> {
  let summary = "ShiftRightLogical operation";
  let description = [{
    Performs element-wise logical right-shift operation on the `lhs` tensor by
    `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical

    Example:
    ```mlir
    %result = stablehlo.shift_right_logical %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def StableHLO_SubtractOp : StableHLO_BinaryElementwiseOp<"subtract",
      [Pure, HLO_CompatibleOperandsAndResultType], HLO_IntFpOrComplexTensor> {
  let summary = "Subtract operation";
  let description = [{
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract

    Example:
    ```mlir
    %result = stablehlo.subtract %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class StableHLO_BinaryBiwiseOrLogicalElementwiseOp<string mnemonic> :
        StableHLO_BinaryElementwiseOp<mnemonic,
          [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs
  );
}

def StableHLO_AndOp: StableHLO_BinaryBiwiseOrLogicalElementwiseOp<"and"> {
  let summary = "And operation";
  let description = [{
    Performs element-wise AND of two tensors `lhs` and `rhs` and produces a
    `result` tensor

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and

    Example:
    ```mlir
    %result = stablehlo.and %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
}

def StableHLO_OrOp: StableHLO_BinaryBiwiseOrLogicalElementwiseOp<"or"> {
  let summary = "Or operation";
  let description = [{
    Performs element-wise OR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or

    Example:
    ```mlir
    %result = stablehlo.or %lhs, %rhs : tensor<2xi1>
    ```
  }];
}

def StableHLO_XorOp : StableHLO_BinaryBiwiseOrLogicalElementwiseOp<"xor"> {
  let summary = "Xor operation";
  let description = [{
    Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor

    Example:
    ```mlir
    %result = stablehlo.xor %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO communication op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_InfeedOp : StableHLO_Op<"infeed", []> {
  let summary = "Infeed operation";
  let description = [{
    Reads data from the infeed and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#infeed

    Example:
    ```mlir
    %results:2 = "stablehlo.infeed"(%token) {
      infeed_config = ""
    } : (!stablehlo.token) -> (tensor<3x3x3xi32>, !stablehlo.token)
    ```
  }];

  let arguments = (ins
    HLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$infeed_config,
    OptionalAttr<ArrayAttr>:$layout
  );
  let results = (outs Variadic<HLO_StaticShapeTensorOrToken>);
  let hasVerifier = 1;
}

def StableHLO_OutfeedOp : StableHLO_Op<"outfeed",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Outfeed operation";
  let description = [{
    Writes `inputs` to the outfeed and produces a `result` token.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#outfeed

    Example:
    ```mlir
    %result = "stablehlo.outfeed"(%input0, %token) {
      outfeed_config = ""
    } : (tensor<3x3x3xi32>, !stablehlo.token) -> !stablehlo.token
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    HLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$outfeed_config
  );
  let results = (outs HLO_Token);
}

def StableHLO_SendOp : StableHLO_Op<"send",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Send operation";
  let description = [{
    Sends `inputs` to a channel `channel_id` and produces a `result` token.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#send

    Example:
    ```mlir
    %result = "stablehlo.send"(%operand, %token) {
      // channel_id = 5 : i64,
      // channel_type = #stablehlo<channel_type DEVICE_TO_HOST>,
      channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>,
      is_host_transfer = true
    } : (tensor<3x4xi32>, !stablehlo.token) -> !stablehlo.token
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    HLO_Token:$token,
    StableHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs HLO_Token);
}

def StableHLO_RecvOp : StableHLO_Op<"recv", []> {
  let summary = "Recv operation";
  let description = [{
    Receives data from a channel with `channel_id` and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#recv

    Example:
    ```mlir
    %results:2 = "stablehlo.recv"(%token) {
      // channel_id = 5 : i64,
      // channel_type = #stablehlo<channel_type HOST_TO_DEVICE>,
      channel_handle = #stablehlo.channel_handle<handle = 5, type = 3>,
      is_host_transfer = true
    } : (!stablehlo.token) -> (tensor<3x4xi32>, !stablehlo.token)
    ```
  }];

  let arguments = (ins
    HLO_Token:$token,
    StableHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs Variadic<HLO_StaticShapeTensorOrToken>);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StableHLO parallelism related op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_ReplicaIdOp : StableHLO_Op<"replica_id", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ReplicaId operation";
  let description = [{
    Produces `replica_id` of the current process.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#replica_id

    Example:
    ```mlir
    %result = stablehlo.replica_id : tensor<ui32>
    ```
  }];
  let results = (outs TensorOf<[UI32]>);

  let assemblyFormat = "attr-dict `:` type(results)";
}

def StableHLO_PartitionIdOp : StableHLO_Op<"partition_id", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "PartitionId operation";
  let description = [{
    Produces `partition_id` of the current process.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#partition_id

    Example:
    ```mlir
    %result = stablehlo.partition_id : tensor<ui32>
    ```
  }];
  let results = (outs TensorOf<[UI32]>);

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// StableHLO control flow op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_AfterAllOp : StableHLO_Op<"after_all", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AfterAll operation";
  let description = [{
    Ensures that the operations producing the `inputs` are executed before any
    operations that depend on `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

    Example:
    ```mlir
    %result = stablehlo.after_all %input0, %input1 : !stablehlo.token
    ```
  }];

  let arguments = (ins Variadic<HLO_Token>:$inputs);
  let results = (outs HLO_Token:$result);

  let assemblyFormat = [{
    $inputs attr-dict
      `:` custom<VariadicSameOperandsAndResultType>(ref($inputs), type($inputs), type($result))
  }];
}

// Xla Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. IfOp maps to predicated
// conditional use of kConditional HLO.
def StableHLO_IfOp: StableHLO_Op<"if", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ReturnOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "If operation";
  let description = [{
    Produces the output from executing exactly one branch from `true_branch` or
    `false_branch` depending on the value of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if

    Example:
    %result = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%result_true_branch) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  }];

  let arguments = (ins
    HLO_PredTensor:$pred /*if_i1*/
  );

  let regions = (region SizedRegion<1>:$true_branch /*if_i2*/,
                        SizedRegion<1>:$false_branch /*if_i3*/);

  let results = (outs Variadic<HLO_TensorOrToken>);
}

// Xla Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. CaseOp maps to indexed
// conditional use of kConditional HLO.
def StableHLO_CaseOp: StableHLO_Op<"case", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*case_c4*/
    ]> {
  let summary = "Case operation";
  let description = [{
    Produces the output from executing exactly one `function` from `branches`
    depending on the value of `index`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case

    Example:
    ```mlir
    %result0, %result1 = "stablehlo.case"(%index) ({
      stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
    }, {
      stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    ```
  }];

  let arguments = (ins
    I32Tensor:$index /*case_i1*/
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches /*case_i2*/);

  let results = (outs Variadic<HLO_TensorOrToken>);
}


def StableHLO_WhileOp: StableHLO_Op<"while", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface>,
      OpAsmOpInterface
    ]> {
  let summary = "While operation";
  let description = [{
    Produces the output from executing `body` function 0 or more times while the
    `cond` function outputs `true`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while

    Example:
    ```mlir
    %results0, %results1 = "stablehlo.while"(%operand0, %operand1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "stablehlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction LT>
        } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "stablehlo.return"(%0) : (tensor<i1>) -> ()
    }, {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "stablehlo.add"(%arg0, %constant0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "stablehlo.return"(%0, %arg1) : (tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    ```
  }];
  let arguments = (ins Variadic<HLO_TensorOrToken>:$operand);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let results = (outs Variadic<HLO_TensorOrToken>);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Method of OpAsmOpInterface used during custom printing to name the block
    // arguments in the nested regions. We name both the condition and the body
    // regions entry arguments the same way, with a `iterArg` prefix. Since the
    // two regions are side-by-side they will have the same name, which allows
    // us to print them once and share it for the two regions, and still be able
    // to parse them back.
    void getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn setNameFn) {
      for (BlockArgument arg : region.getArguments())
        setNameFn(arg, "iterArg");
    }

    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_AllGatherOp : StableHLO_Op<"all_gather", [SameOperandsAndResultElementType]> {
  string summary = "AllGather operation";
  string description = [{
    Within each process group in the process grid, concatenates the values of the
    `operand` tensor from each process along `all_gather_dim` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_gather

    Example:
    ```mlir
    %result = "stablehlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
      // channel_id = 0
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
      // use_global_device_ids = false
    } : (tensor<2x2xf32>) -> tensor<2x4xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;
}

def StableHLO_AllReduceOp : StableHLO_Op<"all_reduce",
    [HLO_CompatibleOperandsAndResultType]> {
  let summary = "AllReduce operation";
  let description = [{
    Within each process group in the process grid, applies a reduction function
    `computation` to the values of the `operand` tensor from each process and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_reduce

    Example:
    ```mlir
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %0 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %0 : tensor<f32>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
      // channel_id = 0
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
      // use_global_device_ids = false
    } : (tensor<4xf32>) -> tensor<4xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;
}

def StableHLO_ReduceScatterOp : StableHLO_Op<"reduce_scatter",
    [SameOperandsAndResultElementType]> {
  let summary = "ReduceScatter operation";
  let description = [{
     Within each process group in the process grid, performs reduction, using
     `computations`, over the values of the `operand` tensor from each process,
     splits the reduction result along `scatter_dimension` into parts, and
     scatters the split parts between the processes to produce the `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_scatter

    Example:
    ```mlir
    %result = "stablehlo.reduce_scatter"(%operand) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %0 : tensor<f32>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      // channel_id = 0
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
      // use_global_device_ids = false
    } : (tensor<2x4xf32>) -> tensor<2x2xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$scatter_dimension,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;
}

def StableHLO_AllToAllOp : StableHLO_Op<"all_to_all",
    [SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "AllToAll operation";
  let description = [{
    Within each process group in the process grid, splits the values of the
    `operand` tensor along `split_dimension` into parts, scatters the split parts
    between the processes, concatenates the scattered parts along `concat_dimension`
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_to_all

    Example:
    ```mlir
    %result = "stablehlo.all_to_all"(%operand) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4xf32>) -> tensor<4x2xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$split_dimension,
    I64Attr:$concat_dimension,
    I64Attr:$split_count,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs HLO_Tensor);

  // channel_handle is only used for the SPMD partitioner, so we add a
  // simplified builder method for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::IntegerAttr": $split_dimension,
      "::mlir::IntegerAttr": $concat_dimension,
      "::mlir::IntegerAttr": $split_count,
      "::mlir::DenseIntElementsAttr": $replica_groups)>];
}

def StableHLO_ReduceOp: StableHLO_ShapedInterfaceOp<"reduce", [
      RecursiveMemoryEffects,
      SameVariadicOperandSize,
      InferTensorTypeWithReify,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]> {
  let summary = "Reduce operation";
  let description = [{
    Applies a reduction function `body` to `inputs` and `init_values` along the
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "stablehlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      dimensions = dense<1> : tensor<1xi64>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
    ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    Variadic<HLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );

  let results = (outs Variadic<HLO_Tensor>);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);
}

//===----------------------------------------------------------------------===//
// StableHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def StableHLO_GetTupleElementOp: StableHLO_Op<"get_tuple_element", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "GetTupleElement operation";
  let description = [{
    Extracts element at `index` position of the `operand` tuple and produces a
    `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_tuple_element

    Example:
    ```mlir
    %result = stablehlo.get_tuple_element %operand[0] : (tuple<tensor<2xf32>, tuple<tensor<i32>>>) -> tensor<2xf32>
    ```
  }];
  let arguments = (ins
    HLO_Tuple:$operand,
    I32Attr:$index
  );

  let results = (outs HLO_TensorOrTokenOrTuple);

  let assemblyFormat = [{
    $operand `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_TupleOp : StableHLO_Op<"tuple", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from values `val`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tuple

    Example:
    ```mlir
    %result = stablehlo.tuple %val0, %val1 : tuple<tensor<2xf32>, tuple<tensor<i32>>>
    ```
   }];

  let arguments = (ins Variadic<HLO_TensorOrTokenOrTuple>:$val);
  let results = (outs HLO_Tuple:$result);

  let assemblyFormat = [{
    $val attr-dict `:` custom<TupleOpType>(type($val), type($result))
  }];
}

def StableHLO_CompareOp: StableHLO_Op<"compare", [Pure, Elementwise,
    SameOperandsElementType /*compare_c1*/,
    SameOperandsAndResultShape /*compare_c2*/,
    InferTensorTypeWithReify /*compare_c1, compare_c2*/]> {
  let summary = "Compare operation";
  let description = [{
    Performs element-wise comparison of `lhs` and `rhs` tensors according to
    `comparison_direction` and `compare_type`, and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare

    Example:
    ```mlir
    %result = stablehlo.compare LT, %lhs, %rhs, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$lhs /*compare_i1*/,
    HLO_Tensor:$rhs /*compare_i2*/,
    StableHLO_ComparisonDirectionAttr:$comparison_direction /*compare_i3*/,
    OptionalAttr<StableHLO_ComparisonTypeAttr>:$compare_type /*compare_i4*/
  );
  let results = (outs HLO_PredTensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
      "::mlir::stablehlo::ComparisonDirection":$comparison_direction,
      CArg<"::mlir::stablehlo::ComparisonType",
      "::mlir::stablehlo::ComparisonType::NOTYPE">:$compare_type)>,
  ];

  let assemblyFormat = [{
    $comparison_direction `,` $lhs `,` $rhs (`,` $compare_type^)?
      attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO Slice definitions.
//===----------------------------------------------------------------------===//

def StableHLO_SliceOp: StableHLO_Op<
      "slice",
      [Pure, SameOperandsAndResultElementType /*slice_c1*/,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]> /*slice_c2*/,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from the `operand` using statically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = "stablehlo.slice" (%operand) {
      start_indices = dense<[1, 2]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi64>) -> tensor<2x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );

  let results = (outs HLO_Tensor);
}

def StableHLO_DynamicSliceOp: StableHLO_Op<"dynamic_slice",
      [Pure, AllElementTypesMatch<["operand", "result"]> /*dynamic_slice_c1*/,
       InferTensorType]> {
  let summary = "DynamicSlice operation";
  let description = [{
    Extracts a slice from the `operand` using dynamically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

    Example:
    ```mlir
    %result = stablehlo.dynamic_slice %operand, %start_indices0, %start_indices1, sizes = [2, 2]
      : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_slice_i1*/,
    Variadic<HLO_ScalarIntTensor>:$start_indices /*dynamic_slice_i2*/,
    I64ElementsAttr:$slice_sizes /*dynamic_slice_i3*/
  );

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $operand `,` custom<VariadicOperandWithAttribute>($start_indices)
      `sizes` `=` custom<DenseI64Array>($slice_sizes)
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_DynamicUpdateSliceOp: StableHLO_Op<"dynamic_update_slice",
      [Pure, AllElementTypesMatch<["operand", "update", "result"]> /*dynamic_update_slice_c1, dynamic_update_slice_c2*/,
       InferTensorType]> {
  let summary = "DynamicUpdateSlice operation";
  let description = [{
    Produces a `result` tensor which is equal to the `operand` tensor except
    that the slice starting at `start_indices` is updated with the values in
    `update`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_update_slice

    Example:
    ```mlir
    %result = stablehlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1
      : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_update_slice_i1*/,
    HLO_Tensor:$update /*dynamic_update_slice_i2*/,
    Variadic<HLO_ScalarIntTensor>:$start_indices /*dynamic_update_slice_i3*/
  );
  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}


//===----------------------------------------------------------------------===//
// StableHLO Other op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_BatchNormGradOp : StableHLO_Op<"batch_norm_grad", [Pure,
    AllElementTypesMatch<["operand", "grad_operand", "grad_scale", "grad_offset"]>,
    InferTensorType]> {
  let summary = "BatchNormGrad operation";
  let description = [{
    Computes gradients of several inputs of BatchNormTrainingOp backpropagating
    from `grad_output`, and produces `grad_operand`, `grad_scale` and
    `grad_offset` tensors.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_grad

    Example:
    ```mlir
    %grad_operand, %grad_scale, %grad_offset =
    "stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
      epsilon = 0.0 : f32,
      feature_index = 2 : i64
    } : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2x2x2xf32>) -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    ```
  }];

  let arguments = (ins
    RankedTensorOf<[HLO_Float]>:$operand,
    1DTensorOf<[HLO_Float]>:$scale,
    1DTensorOf<[HLO_Float]>:$mean,
    1DTensorOf<[HLO_Float]>:$variance,
    RankedTensorOf<[HLO_Float]>:$grad_output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
      RankedTensorOf<[HLO_Float]>:$grad_operand,
      1DTensorOf<[HLO_Float]>:$grad_scale,
      1DTensorOf<[HLO_Float]>:$grad_offset);
}

def StableHLO_BatchNormInferenceOp : StableHLO_Op<"batch_norm_inference",
    [Pure, AllElementTypesMatch<["operand", "result"]>, InferTensorType]> {
  let summary = "BatchNormInference operation";
  let description = [{
    Normalizes the `operand` tensor across all dimensions except for the
    `feature_index` dimension and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_inference

    Example:
    ```mlir
    %result = "stablehlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) {
      epsilon = 0.0 : f32,
      feature_index = 2 : i64
    } : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x2xf32>
    ```
  }];

  let arguments = (ins
    RankedTensorOf<[HLO_Float]>:$operand,
    1DTensorOf<[HLO_Float]>:$scale,
    1DTensorOf<[HLO_Float]>:$offset,
    1DTensorOf<[HLO_Float]>:$mean,
    1DTensorOf<[HLO_Float]>:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs RankedTensorOf<[HLO_Float]>:$result);
}

def StableHLO_BatchNormTrainingOp : StableHLO_Op<"batch_norm_training",
    [Pure, AllElementTypesMatch<["operand", "output", "batch_mean", "batch_var"]>,
    InferTensorType]> {
  let summary = "BatchNormTraining operation";
  let description = [{
    Computes mean and variance across batch and spatial dimensions and
    normalizes the `operand` tensor, for each feature in the `feature_index`
    dimension and produces `output`, `batch_mean` and `batch_var` tensors.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_training

    Example:
    ```mlir
    %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%operand, %scale, %offset) {
      epsilon = 0.0 : f32,
      feature_index = 2 : i64
    } : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    ```
  }];

  let arguments = (ins
    RankedTensorOf<[HLO_Float]>:$operand,
    1DTensorOf<[HLO_Float]>:$scale,
    1DTensorOf<[HLO_Float]>:$offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
      RankedTensorOf<[HLO_Float]>:$output,
      1DTensorOf<[HLO_Float]>:$batch_mean,
      1DTensorOf<[HLO_Float]>:$batch_var);
}

def StableHLO_BitcastConvertOp : StableHLO_ShapedInterfaceOp<"bitcast_convert",
    [Pure]> {
  let summary = "BitcastConvert operation";
  let description = [{
    Performs a bitcast operation on `operand` tensor and produces a `result`
    tensor where the bits of the entire `operand` tensor are reinterpreted using
    the type of the `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert

    Example:
    ```mlir
    %result = stablehlo.bitcast_convert %operand : (tensor<2xf32>) -> tensor<2x4xi8>
    ```
  }];

  let arguments = (ins HLO_Tensor:$operand);
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_BroadcastOp : StableHLO_ShapedInterfaceOp<"broadcast",
    [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "Broadcast operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as XLA's Broadcast:
    https://www.tensorflow.org/xla/operation_semantics#broadcast

    Example:
    ```mlir
    %result = stablehlo.broadcast %operand, sizes = [1, 2] : (tensor<3xi32>) -> tensor<1x2x3xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` `sizes` `=` custom<DenseI64Array>($broadcast_sizes)
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_BroadcastInDimOp : StableHLO_Op<"broadcast_in_dim",
      [Pure, SameOperandsAndResultElementType /*broadcast_in_dim_c1*/]> {
  let summary = "BroadcastInDim operation";
  let description = [{
    Expands the dimensions and/or rank of an input tensor by duplicating the
    data in the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    Example:
    ```mlir
    %result = stablehlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*broadcast_in_dim_i1*/,
    BroadcastDimAttr:$broadcast_dimensions /*broadcast_in_dim_i2*/
  );

  let results = (outs HLO_StaticShapeTensor);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` `dims` `=` custom<DenseI64Array>($broadcast_dimensions)
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_DynamicBroadcastInDimOp : StableHLO_ShapedInterfaceOp<
    "dynamic_broadcast_in_dim", [Pure]> {
  let summary = "DynamicBroadcastInDim operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as BroadcastInDimOp except
    that the result shape is specified dynamically via `output_dimensions`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    It also accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_DimensionTensor:$output_dimensions,
    BroadcastDimAttr:$broadcast_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_expanding_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_nonexpanding_dimensions
  );

  let results = (outs HLO_Tensor);

  let builders = [
    OpBuilder<(ins
        "Type":$result_type, "Value":$operand, "Value":$output_dimensions,
        "DenseIntElementsAttr":$broadcast_dimensions), [{
      build($_builder, $_state, result_type, operand, output_dimensions,
          broadcast_dimensions, /*known_expanding_dimensions=*/{},
          /*known_nonexpanding_dimensions=*/{});
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` $output_dimensions `,` `dims` `=` custom<DenseI64Array>($broadcast_dimensions)
      attr-dict `:` functional-type(operands, results)
  }];
}

// Note: There is no HLO_CallOp because the standard call operation mlir::func::CallOp
// is used instead. A mlir::func::CallOp is exported to a HLO call instruction
// directly.

def StableHLO_CholeskyOp : StableHLO_Op<"cholesky",
      [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "Cholesky operation";
  let description = [{
    Computes the Cholesky decomposition of a batch of matrices.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cholesky

    Example:
    ```mlir
    %result = stablehlo.cholesky %a, lower = true : tensor<3x3xf32>
    ```
  }];
  let arguments = (ins
    HLO_FpOrComplexTensor:$a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$lower
  );

  let results = (outs HLO_FpOrComplexTensor:$result);

  let assemblyFormat = [{
    $a (`,` `lower` `=` $lower^)? attr-dict `:` custom<SameOperandsAndResultType>(type($a), type($result))
  }];
}

def StableHLO_ClampOp : StableHLO_ShapedInterfaceOp<"clamp", [Pure,
  SameOperandsAndResultElementType /* clamp_c3 */, HLO_BroadcastingElementwise,
  InferTensorType]> {
  let summary = "Clamp operation";
  let description = [{
    Clamps every element of the `operand` tensor between a minimum and maximum
    value and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp

    Example:
    ```mlir
    %result = stablehlo.clamp %min, %operand, %max : tensor<3xi32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$min, /*clamp_i1*/
    HLO_Tensor:$operand, /*clamp_c3, clamp_i2*/
    HLO_Tensor:$max /*clamp_i3*/
  );
  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $min `,` $operand `,` $max attr-dict
      `:` custom<SameOperandsAndResultType>(type($min), type($operand), type($max), type($result))
  }];
}

def StableHLO_ConcatenateOp : StableHLO_ShapedInterfaceOp<"concatenate",
    [Pure, SameOperandsAndResultElementType /*concatenate_c1, concatenate_c3, concatenate_c5*/,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Concatenate operation";
  let description = [{
    Concatenates a variadic number of tensors in `inputs` along `dimension`
    dimension in the same order as the given arguments and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*concatenate_i1*/,
    I64Attr:$dimension /*concatenate_i2*/
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
     custom<VariadicOperandWithAttribute>($inputs) `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_CollectivePermuteOp: StableHLO_Op<"collective_permute",
    [HLO_CompatibleOperandsAndResultType]> {
  let summary = "CollectivePermute operation";
  let description = [{
    Within each process group in the process grid, sends the value of the
    `operand` tensor from the source process to the target process and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective_permute

    Example:
    ```mlir
    %result = "stablehlo.collective_permute"(%operand) {
      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
      // channel_id = 0
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4x2xf32>) -> tensor<4x2xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$source_target_pairs,
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;
  // channel_handle is only used for the SPMD partitioner, so we add a
  // simplified builder method for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::DenseIntElementsAttr":$source_target_pairs)>];
}

def StableHLO_ConvolutionOp : StableHLO_Op<"convolution", [Pure]> {
  let summary = "Convolution operation";
  let description = [{
    Computes dot products between windows of `lhs` and slices of `rhs` and
    produces `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution

    Example:
    ```mlir
    %result = "stablehlo.convolution"(%lhs, %rhs) {
      window_strides = dense<4> : tensor<2xi64>,
      padding = dense<0> : tensor<2x2xi64>,
      lhs_dilation = dense<2> : tensor<2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_reversal = dense<false> : tensor<2xi1>,
      dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x4x4x1xi32>, tensor<3x3x1x1xi32>) -> tensor<1x2x2x1xi32>
    ```
  }];
  let arguments = !con(
    (ins
       HLO_Tensor:$lhs,
       HLO_Tensor:$rhs),
    StableHLO_ConvolutionAttributes.attributes);

  let results = (outs HLO_Tensor);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasWindowReversal() {
      auto reversal = getWindowReversalAttr();
      return reversal && llvm::any_of(reversal.getValues<bool>(),
                                      [](bool v) { return v; });
    }
  }];

  let assemblyFormat = [{
    `(`operands`)`
       `dim_numbers` `=` custom<ConvolutionDimensions>($dimension_numbers) `,`
       `window` `=` `{` custom<WindowAttributes>($window_strides, $padding,
                                                 $lhs_dilation, $rhs_dilation,
                                                 $window_reversal) `}`
       attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_CrossReplicaSumOp : StableHLO_Op<"cross-replica-sum",
    [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "CrossReplicaSum operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as AllReduceOp with
    `channel_id = 0`, `use_global_device_ids = false` and `computation`
    implementing addition:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_reduce

    Example:
    ```mlir
    %result = "stablehlo.cross-replica-sum"(%operand) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<4xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );

  let results = (outs HLO_Tensor);
}

def StableHLO_CustomCallOp: StableHLO_Op<"custom_call",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "CustomCall operation";
  let description = [{
    Encapsulates an implementation-defined operation `call_target_name` that
    takes `inputs` and `called_computations` and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call

    Example:
    ```mlir
    %results = "stablehlo.custom_call"(%input0) {
      call_target_name = "foo",
      has_side_effect = false,
      backend_config = "bar",
      api_version = 1 : i32,
      called_computations = [@foo]
    } : (tensor<f32>) -> tensor<f32>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrTokenOrTuple>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    // TODO(b/189822916): Remove this field when all clients are migrated to
    // the status-returning API.
    DefaultValuedOptionalAttr<
        StableHLO_CustomCallApiVersionAttr,
        "::mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
        $api_version,
    DefaultValuedOptionalAttr<StableHLO_FlatSymbolRefArrayAttr, "{}">:$called_computations,
    OptionalAttr<StableHLO_ArrayOfLayoutAttr>:$operand_layouts,
    OptionalAttr<StableHLO_ArrayOfLayoutAttr>:$result_layouts,
    DefaultValuedOptionalAttr<
        TypedArrayAttrBase<
            StableHLO_OutputOperandAlias,
            "Aliasing attribute for outputs and operands of CustomCall">,
        "{}">:$output_operand_aliases
  );
  let results = (outs Variadic<HLO_TensorOrTokenOrTuple>);
  let hasVerifier = 1;

  let assemblyFormat = [{
    custom<CustomCallTarget>($call_target_name) `(` $inputs `)`
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_DotOp: StableHLO_Op<"dot", [Pure]> {
  let summary = "Dot operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as XLA's Dot:
    https://www.tensorflow.org/xla/operation_semantics#dot

    Example:
    ```mlir
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<1x1xi32>
    ```
  }];
  let arguments = (
    ins HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    StableHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;

  // Use empty `` to prevent extra whitespace before precision config.
  let assemblyFormat = [{
    $lhs `,` $rhs `` custom<PrecisionConfig>($precision_config) attr-dict
      `:` functional-type(operands, results)
  }];
}

def StableHLO_DotGeneralOp: StableHLO_ShapedInterfaceOp<"dot_general", [Pure]> {
  let summary = "DotGeneral operation";
  let description = [{
    Computes dot products between slices of `lhs` and slices of `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general

    Example:
    ```mlir
    %result = stablehlo.dot_general %lhs, %rhs,
      batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT]
      : (tensor<2x2x2xi32>, tensor<2x2x2xi32>) -> tensor<2x2x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    StableHLO_DotDimensionNumbers:$dot_dimension_numbers,
    StableHLO_PrecisionConfigAttr:$precision_config
  );

  let results = (outs HLO_Tensor);
  let hasVerifier = 1;

  // Use empty `` to prevent extra whitespace before precision config.
  let assemblyFormat = [{
    $lhs `,` $rhs `,` custom<DotDimensionNumbers>($dot_dimension_numbers) ``
    custom<PrecisionConfig>($precision_config) attr-dict
      `:` functional-type(operands, results)
  }];
}

def StableHLO_EinsumOp: StableHLO_Op<"einsum", [Pure]> {
  let summary = "Einsum operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as TF's einsum:
    https://www.tensorflow.org/api_docs/python/tf/einsum

    Example:
    ```mlir
    %result = "stablehlo.einsum"(%lhs, %rhs) {
      einsum_config = "ab,bc->ac"
    } : (tensor<4x16xf32>, tensor<16x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` `config` `=` $einsum_config attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_UnaryEinsumOp: StableHLO_Op<"unary_einsum", [Pure]> {
  let summary = "UnaryEinsum operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as TF's einsum:
    https://www.tensorflow.org/api_docs/python/tf/einsum

    Example:
    ```mlir
    %result = "stablehlo.unary_einsum"(%operand) {
      einsum_config = "ab->a"
    } : (tensor<4x16xf32>) -> tensor<4xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    StrAttr:$einsum_config
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` `config` `=` $einsum_config attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_FftOp: StableHLO_Op<"fft", [InferTensorType, Pure]> {
  let summary = "Fft operation";
  let description = [{
    Performs the forward and inverse Fourier transforms for real and complex
    inputs/outputs.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#fft

    Example:
    ```mlir
    %result = stablehlo.fft %operand, type = FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
    ```
  }];
  let arguments = (ins
    HLO_FpOrComplexTensor:$operand,
    StableHLO_FftTypeAttr:$fft_type,
    I64ElementsAttr:$fft_length
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` `type` `=` $fft_type `,` `length` `=` custom<DenseI64Array>($fft_length)
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_GatherOp: StableHLO_Op<"gather", [InferTensorTypeWithReify, Pure]> {
  let summary = "Gather operation";
  let description = [{
    Gathers slices from `operand` tensor from offsets specified in
    `start_indices` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

    Example:
    ```mlir
    %result = "stablehlo.gather"(%operand, %start_indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2, 3],
        collapsed_slice_dims = [0],
        start_index_map = [0, 2],
        index_vector_dim = 2>,
      slice_sizes = dense<[0, 2, 2]> : tensor<3xi64>,
      indices_are_sorted = false
    } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_IntTensor:$start_indices,
    StableHLO_GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );

  let results = (outs HLO_Tensor);
}

def StableHLO_GetDimensionSizeOp: StableHLO_Op<"get_dimension_size",
      [Pure, InferTensorType]> {
  let summary = "GetDimensionSize operation";
  let description = [{
    Produces the size of the given `dimension` of the `operand`.

    See
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_dimension_size

    Example:
    ```mlir
    %result = stablehlo.get_dimension_size %operand, dim = 1 : (tensor<2x3xf32>) -> tensor<i32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$dimension
  );
  // TODO(hinsu): Allow 64-bit result types once XLA HLO dialect based on the
  // XLA semantics is available. This limitation is because of the current XLA
  // implementation.
  let results = (outs I32Tensor);

  let assemblyFormat = [{
    $operand `,` `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_MapOp: StableHLO_ShapedInterfaceOp<"map",
      [RecursiveMemoryEffects, SameOperandsAndResultShape,
       SingleBlockImplicitTerminator<"ReturnOp">, InferTensorTypeWithReify]> {
  let summary = "Map operation";
  let description = [{
    Applies a map function `computation` to `inputs` along the `dimensions` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#map

    Example:
    ```mlir
    %result = "stablehlo.map"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = stablehlo.multiply %arg0, %arg1 : tensor<i32>
        stablehlo.return %0 : tensor<i32>
    }) {
      dimensions = dense<[0, 1]> : tensor<2xi64>
    } : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs HLO_Tensor);
}

def StableHLO_ReshapeOp: StableHLO_Op<"reshape",
      [Pure, SameOperandsAndResultElementType]> {
  let summary = "Reshape operation";
  let description = [{
    Performs reshape of `operand` tensor to a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %result = stablehlo.reshape %operand : (tensor<2xf32>) -> tensor<1x2xf32>
    ```
  }];

  let arguments = (ins HLO_Tensor:$operand);

  let results = (outs HLO_StaticShapeTensor);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_DynamicReshapeOp: StableHLO_ShapedInterfaceOp<"dynamic_reshape", [Pure]> {
  let summary = "DynamicReshape operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as ReshapeOp except that the
    result shape is specified dynamically via `output_shape`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %0 = stablehlo.dynamic_reshape %arg0, %shape : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    ```
  }];

  let arguments = (ins HLO_Tensor:$operand, HLO_DimensionTensor:$output_shape);
  let results = (outs HLO_Tensor:$result);

  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_ScatterOp: StableHLO_Op<"scatter",
      [SameVariadicOperandSize, RecursiveMemoryEffects,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Scatter operation";
  let description = [{
    Produces `results` tensors which are equal to `inputs` tensors except that
    several slices specified by `scatter_indices` are updated with the values
    `updates` using `update_computation`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

   Example:

   ```mlir
   %result = "stablehlo.scatter"(%input, %scatter_indices, %update) ({
     ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
       %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
       stablehlo.return %0 : tensor<i32>
   }) {
     scatter_dimension_numbers = #stablehlo.scatter<
       update_window_dims = [2,3],
       inserted_window_dims = [0],
       scatter_dims_to_operand_dims = [1, 0],
       index_vector_dim = 2>,
     indices_are_sorted = false,
     unique_indices = false
   } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<2x3x2x2xi32>) -> tensor<3x4x2xi32>
   ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    TensorOf<[AnyInteger, Index]>:$scatter_indices,
    Variadic<HLO_Tensor>:$updates,
    StableHLO_ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unique_indices
  );

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs Variadic<HLO_Tensor>);

  let hasVerifier = 1;
}

def StableHLO_SelectOp: StableHLO_Op<"select", [Pure, HLO_BroadcastingElementwise,
    InferTensorTypeWithReify]> {
  let summary = "Select operation";
  let description = [{
    Produces a `result` tensor where each element is selected from `on_true` or
    `on_false` tensor based on the value of the corresponding element of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select

    Example:
    ```mlir
    %result = stablehlo.select %pred, %on_true, %on_false : tensor<2x2xi1>, tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_PredTensor:$pred, /*select_i1*/
    HLO_Tensor:$on_true, /*select_i2*/
    HLO_Tensor:$on_false /*select_i3*/
  );

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    operands attr-dict `:`
      custom<SelectOpType>(type($pred), type($on_true), type($on_false), type($result))
  }];
}

def StableHLO_SelectAndScatterOp: StableHLO_Op<"select_and_scatter",
      [RecursiveMemoryEffects, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "SelectAndScatter operation";
  let description = [{
    Scatters the values from the `source` tensor using `scatter` based on the
    outcome of `reduce_window` of the `input` tensor using `select` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select_and_scatter

    Example:
    ```mlir
    %result = "stablehlo.select_and_scatter"(%operand, %source, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "stablehlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction GE>
        } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "stablehlo.return"(%0) : (tensor<i1>) -> ()
    }, {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "stablehlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      window_dimensions = dense<[3, 1]> : tensor<2xi64>,
      window_strides = dense<[2, 1]> : tensor<2xi64>,
      padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<4x2xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$source,
    HLO_Tensor:$init_value,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);

  let results = (outs HLO_Tensor);

  let hasVerifier = 1;
}

def StableHLO_SetDimensionSizeOp: StableHLO_Op<"set_dimension_size", [Pure,
      InferTensorType]> {
  let summary = "SetDimensionSize operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as XLA's SetDimensionSize:
    https://www.tensorflow.org/xla/operation_semantics#setdimensionsize

    Example:
    ```mlir
    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I32Tensor:$size,
    I64Attr:$dimension
  );
  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` $size  `,` `dim` `=` $dimension attr-dict
      `:` functional-type(operands, results)
  }];
}

def StableHLO_SortOp : StableHLO_Op<"sort",
      [RecursiveMemoryEffects, SameOperandsAndResultShape, InferTensorType]> {
  let summary = "Sort operation";
  let description = [{
    Sorts a variadic number of tensors in `inputs` together, according to a
    custom `comparator`, along the given `dimension` and produces a variadic
    number of tensors as `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sort

    Example:
    ```mlir
    %result0, %result1 = "stablehlo.sort"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
        %predicate = "stablehlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction GT>
          } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "stablehlo.return"(%predicate) : (tensor<i1>) -> ()
    }) {
      dimension = 0 : i64,
      is_stable = true
    } : (tensor<2x3xi32>, tensor<2x3xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>)
    ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable
  );

  let results = (outs Variadic<HLO_Tensor>);

  let regions = (region SizedRegion<1>:$comparator);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, CArg<"int64_t", "-1">:$dimension,
      CArg<"bool", "false">:$is_stable)>];

  let hasVerifier = 1;
}

def StableHLO_ReverseOp: StableHLO_Op<"reverse",
      [Pure, HLO_CompatibleOperandsAndResultType /*reverse_c1*/]> {
  let summary = "Reverse operation";
  let description = [{
    Reverses the order of elements in the `operand` along the specified
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reverse

    Example:
    ```mlir
    %result = stablehlo.reverse %operand, dims = [1] : tensor<3x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );

  let hasVerifier = 1;

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $operand `,` `dims` `=` custom<DenseI64Array>($dimensions)
      attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def StableHLO_PadOp: StableHLO_ShapedInterfaceOp<"pad",
      [Pure, SameOperandsAndResultElementType /*pad_c1*/,
      AllTypesMatch<["edge_padding_low", "edge_padding_high",
      "interior_padding"]> /*pad_c2, pad_i3, pad_i4, pad_i5*/,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Pad operation";
  let description = [{
    Expands `operand` by padding around the tensor as well as between the
    elements of the tensor with the given `padding_value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

    Example:
    ```mlir
    %0 = stablehlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1], interior = [1, 2]
      : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*pad_i1*/,
    HLO_Tensor:$padding_value /*pad_i2*/,
    I64ElementsAttr:$edge_padding_low /*pad_i3*/,
    I64ElementsAttr:$edge_padding_high /*pad_i4*/,
    I64ElementsAttr:$interior_padding /*pad_i5*/
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` $padding_value `,`
      `low` `=` custom<DenseI64Array>($edge_padding_low) `,`
      `high` `=` custom<DenseI64Array>($edge_padding_high) `,`
      `interior` `=` custom<DenseI64Array>($interior_padding)
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_TraceOp: StableHLO_Op<"trace", []> {
  let summary = "Trace operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/604.

    It is not used by JAX, PyTorch or TensorFlow, so it looks like we should've
    classified it as "Private to XLA" and not included it in StableHLO in the
    first place. With that in mind, its semantics will not be documented here.

    Example:
    ```mlir
    stablehlo.trace %arg0, "In test code." : tensor<5x1x5xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    StrAttr:$tag
  );
  let assemblyFormat = "$operand `,` $tag attr-dict `:` type($operand)";
}

def StableHLO_TransposeOp: StableHLO_ShapedInterfaceOp<"transpose",
      [Pure, SameOperandsAndResultElementType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Transpose operation";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose

    Example:
    ```mlir
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<1x2x3xi32>) -> tensor<3x2x1xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` `dims` `=` custom<DenseI64Array>($permutation)
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_TriangularSolveOp: StableHLO_Op<"triangular_solve",
    [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "TriangularSolve operation";
  let description = [{
    Solves batches of systems of linear equations with lower or upper triangular
    coefficient matrices.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#triangular_solve

    Example:
    ```mlir
    %result = "stablehlo.triangular_solve"(%a, %b) {
      left_side = true,
      lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    } : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    ```
  }];
  let arguments = (ins
    HLO_FpOrComplexTensor:$a,
    HLO_FpOrComplexTensor:$b,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    StableHLO_TransposeAttr:$transpose_a
  );
  let results = (outs HLO_FpOrComplexTensor);
}

def StableHLO_ReduceWindowOp: StableHLO_Op<"reduce_window", [
      RecursiveMemoryEffects,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">,
      InferTensorType,
    ]> {
  let summary = "ReduceWindow operation";
  let description = [{
    Applies a reduction function `body` to windows of `inputs` and `init_values`
    and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window

    Example:
    ```mlir
    %result = "stablehlo.reduce_window"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
        stablehlo.return %0 : tensor<i32>
    }) {
      window_dimensions = dense<[2, 1]> : tensor<2xi64>,
      window_strides = dense<[4, 1]> : tensor<2xi64>,
      base_dilations = dense<[2, 1]> : tensor<2xi64>,
      window_dilations = dense<[3, 1]> : tensor<2xi64>,
      padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<3x2xi32>, tensor<i32>) -> tensor<2x2xi32>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs,
    Variadic<HLO_Tensor>:$init_values,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the operand dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs Variadic<HLO_Tensor>);

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;

  // Builder for non-variadic version of the operation.
  let builders = [
    OpBuilder<(ins "Type":$result_type, "Value":$operand,
      "Value":$init_value,
      "DenseIntElementsAttr":$window_dimensions,
      "DenseIntElementsAttr":$window_strides,
      "DenseIntElementsAttr":$base_dilations,
      "DenseIntElementsAttr":$window_dilations,
      "DenseIntElementsAttr":$padding),
    [{
      build($_builder, $_state, TypeRange(result_type), ValueRange(operand),
            ValueRange(init_value), window_dimensions, window_strides,
            base_dilations, window_dilations, padding);
    }]>,
    OpBuilder<(ins "ValueRange":$operands,
      "ValueRange":$init_values,
      "DenseIntElementsAttr":$window_dimensions,
      "DenseIntElementsAttr":$window_strides,
      "DenseIntElementsAttr":$base_dilations,
      "DenseIntElementsAttr":$window_dilations,
      "DenseIntElementsAttr":$padding,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$bodyBuilder
    )>,
  ];

  // TODO(hinsu): Implement custom printer and parser.
}

def StableHLO_ReturnOp : StableHLO_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation";
  let summary = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/425.

    Informally, this operation serves as a terminator for regions defined by
    the StableHLO ops. Non-StableHLO ops, e.g. `func.func`, have their own
    terminators, e.g. `func.return`.

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "stablehlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      dimensions = dense<1> : tensor<1xi64>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrTokenOrTuple >:$results
  );

  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

def StableHLO_TorchIndexSelectOp : StableHLO_Op<"torch_index_select", [Pure]> {
  let summary = "TorchIndexSelect operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as PyTorch's index_select,
    augmented with support for batch dimensions:
    https://pytorch.org/docs/stable/generated/torch.index_select.html.

    The `batch_dims` attribute specifies the number of major batch dimensions
    (0 or more) that act like a multidimensional loop over both the operand and
    the index.

    Example:
    ```mlir
    %result = "stablehlo.torch_index_select"(%operand, %index) {
      dim = 2 : i64,
      batch_dims = 1 : i64
    } : (tensor<8x128x3072x64xf32>, tensor<8x16x1024xi32>) -> tensor<8x128x16x1024x64xf32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$index,
    I64Attr:$dim,
    I64Attr:$batch_dims
  );

  let results = (outs HLO_Tensor);
}

def StableHLO_OptimizationBarrierOp : StableHLO_Op<"optimization_barrier",
      [Pure, HLO_PairwiseSameOperandAndResultType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "OptimizationBarrier operation";
  let description = [{
    Ensures that the operations that produce the `operand` are executed before any
    operations that depend on the `result` and prevents compiler transformations
    from moving operations across the barrier. Other than that, the operation is
    an identity, i.e. `result` = `operand`.

    See
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#optimization_barrier

    Example:
    ```mlir
    %result0, %result1 = stablehlo.optimization_barrier %operand0, %operand1 : tensor<f32>, tensor<f32>
    ```
  }];

  let arguments = (ins Variadic<HLO_TensorOrToken>:$operand);

  let results = (outs Variadic<HLO_TensorOrToken>:$result);

  // Use `attr-dict` before `$operand` because Optional Group anchors in custom
  // directives are currently not supported. Also since inputs are variadic,
  // print `()` if no arguments are present, otherwise parsing is ambiguous:
  //   stablehlo.optimization_barrier
  //   %1 = stablehlo.add ...
  //   ^ Without lookahead, ambiguous if this is an operand to the previous line
  //     or the start of a separate operation, since newlines are ignored.
  let assemblyFormat = [{
    attr-dict ($operand^ `:` custom<PairwiseOpType>(type($operand), type($result))):(`(` `)`)?
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO RNG operations.
//===----------------------------------------------------------------------===//

def StableHLO_RngOp : StableHLO_Op<"rng", [InferTensorTypeWithReify, AllElementTypesMatch<["a", "b", "result"]>]> {
  let summary = "Rng operation";
  let description = [{
    Generates random numbers using the `rng_distribution` algorithm and produces
    a `result` tensor of a given shape `shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng

    Example:
    ```mlir
    %result = stablehlo.rng %a, %b, %shape, distribution = NORMAL : (tensor<i32>, tensor<i32>, tensor<2xi64>) -> tensor<3x3xi32>
    ```
  }];
  let arguments = (ins
    0DTensorOf<[HLO_Pred, HLO_Int, HLO_Float]>:$a,
    0DTensorOf<[HLO_Pred, HLO_Int, HLO_Float]>:$b,
    HLO_DimensionTensor:$shape,
    StableHLO_RngDistributionAttr:$rng_distribution
  );

  let results = (outs HLO_PredIntOrFpTensor:$result);

  let assemblyFormat = [{
    $a `,` $b `,` $shape `,` `distribution` `=` $rng_distribution
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_RngBitGeneratorOp : StableHLO_Op<"rng_bit_generator", [Pure]> {
  let summary = "RngBitGenerator operation";
  let description = [{
    Returns an `output` filled with uniform random data and an updated output
    state `output_state` given an initial state `initial_state` using the
    pseudorandom number generator algorithm `rng_algorithm`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng_bit_generator

    Example:
    ```mlir
    %output_state, %output = stablehlo.rng_bit_generator %initial_state, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
    ```
  }];
  let arguments = (ins
    StableHLO_RngAlgorithmAttr:$rng_algorithm,
    HLO_IntOrFpTensor:$initial_state
  );

  let results = (outs
    HLO_IntOrFpTensor:$output_state,
    HLO_StaticShapeIntOrFpTensor:$output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $initial_state `,` `algorithm` `=` $rng_algorithm attr-dict
      `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO Quantize operation.
//===----------------------------------------------------------------------===//

// TODO(b/230662142): Implement unknown scales/zero_point cases.
def StableHLO_UniformQuantizeOp : StableHLO_UnaryElementwiseOp<"uniform_quantize",
      [Pure], TensorOf<[F32, BF16, HLO_QuantizedInt]>,
      HLO_QuantizedIntTensor> {
  let summary = "UniformQuantize operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/588.

    Informally, this operation converts floating point tensors or uniform
    quantized tensors to uniform quantized tensors according to the quantization
    parameters defined by the result type.

    Example:
    ```mlir
    %result = stablehlo.uniform_quantize %operand : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    ```
  }];
}

def StableHLO_UniformDequantizeOp : StableHLO_UnaryElementwiseOp<"uniform_dequantize",
      [InferTensorType, Pure], HLO_QuantizedIntTensor, TensorOf<[F32, BF16]>> {
  let summary = "UniformDequantize operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/588.

    Informally, this operation converts uniform quantized tensors to floating
    point tensors according to the quantization parameters defined by the
    operand type.

    Example:
    ```mlir
    %result = stablehlo.uniform_dequantize %operand : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
    ```
  }];
}

def StableHLO_ReducePrecisionOp :
    StableHLO_Op<"reduce_precision", [HLO_CompatibleOperandsAndResultType, Pure]> {
  let summary = "ReducePrecision operation";
  let description = [{
    Performs element-wise conversion of `operand` to another floating-point type
    that uses `exponent_bits` and `mantissa_bits` and back to the original
    floating-point type and produces an `output` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_precision

    Example:
    ```mlir
    %output = stablehlo.reduce_precision %operand, format = e5m2 : tensor<6xf32>
    ```
  }];
  let arguments = (ins
    HLO_FpTensor:$operand,
    I32Attr:$exponent_bits,
    I32Attr:$mantissa_bits
  );
  let hasVerifier = 1;
  let results = (outs HLO_FpTensor:$output);

  let assemblyFormat = [{
    $operand `,` `format` `=` custom<ExponentMantissa>($exponent_bits, $mantissa_bits)
      attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($output))
  }];
}

def StableHLO_RealDynamicSliceOp: StableHLO_ShapedInterfaceOp<
      "real_dynamic_slice",
      [Pure, AllElementTypesMatch<["operand", "result"]>,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let summary = "RealDynamicSlice operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as SliceOp except
    that `start_indices`, `limit_indices` and `strides` are specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = stablehlo.real_dynamic_slice %operand,
                %start_indices, %limit_indices, %strides
           : (tensor<256x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xf32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_DimensionTensor:$start_indices,
    HLO_DimensionTensor:$limit_indices,
    HLO_DimensionTensor:$strides
  );
  let results = (outs HLO_Tensor:$result);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_DynamicPadOp: StableHLO_ShapedInterfaceOp<"dynamic_pad",
      [Pure, AllElementTypesMatch<["operand", "padding_value", "result"]>,
      AllTypesMatch<["edge_padding_low", "edge_padding_high", "interior_padding"]>]> {
  let summary = "DynamicPad operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as PadOp except
    that `edge_padding_low`, `edge_padding_high` and `interior_padding` are
    specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

    Example:
    ```mlir
    %result = stablehlo.dynamic_pad %operand, %padding_value,
                %edge_padding_low, %edge_padding_high, %interior_padding
           : (tensor<?x?xf32>, tensor<f32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xf32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$padding_value,
    HLO_DimensionTensor:$edge_padding_low,
    HLO_DimensionTensor:$edge_padding_high,
    HLO_DimensionTensor:$interior_padding
  );
  let results = (outs HLO_Tensor:$result);
  let description = [{
    Dynamically Pads the `operand`, with amount of padding added at
    low-end/high-end/interior is passed through input tensors.
  }];
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_DynamicGatherOp: StableHLO_Op<"dynamic_gather",
                                [InferTensorTypeWithReify, Pure]> {
  let summary = "DynamicGather operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as GatherOp except
    that `slice_sizes` are specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

    Example:
    ```mlir
    %result = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2, 3],
        collapsed_slice_dims = [0],
        start_index_map = [0, 2],
        index_vector_dim = 2>,
      indices_are_sorted = false
    } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_IntTensor:$start_indices,
    HLO_IntTensor:$slice_sizes,
    StableHLO_GatherDimensionNumbers:$dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs HLO_Tensor);
}

def StableHLO_DynamicConvOp : StableHLO_Op<"dynamic_conv", [Pure]> {
  let summary = "DynamicConv operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as ConvolutionOp except
    that `padding` is specified dynamically via `d_padding`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution

    Example:
    ```mlir
    %result = "stablehlo.dynamic_conv"(%lhs, %rhs, %d_padding) {
      window_strides = dense<4> : tensor<2xi64>,
      lhs_dilation = dense<2> : tensor<2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_reversal = dense<false> : tensor<2xi1>,
      dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x4x4x1xi32>, tensor<3x3x1x1xi32>, tensor<2x2xi64>) -> tensor<1x2x2x1xi32>
    ```
  }];

  let arguments = !con(
    (ins
       HLO_Tensor:$lhs,
       HLO_Tensor:$rhs,
       HLO_Tensor:$d_padding),
    StableHLO_ConvolutionAttributes.attributes);
  let results = (outs HLO_Tensor);
}

def StableHLO_ComputeReshapeShapeOp :
    StableHLO_Op<"compute_reshape_shape", [Pure]> {
  let summary = "ComputeReshapeShape operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation computes an output_shape for DynamicReshapeOp
    from the `num_elements` number of elements in an operand of DynamicReshapeOp
    and the `dynamic_shape` shape provided to TF's reshape:
    https://www.tensorflow.org/api_docs/python/tf/reshape

    For example, for `num_elements = 12` and `dynamic_shape = [2, -1]`,
    the `result` is going to be `[2, 6]`. If operands are not valid (e.g. if
    dimensions do not evenly divide the number of elements, or if there are
    multiple -1 values in dimensions), this leads to undefined behavior.

    Example:
    ```mlir
    %result = stablehlo.compute_reshape_shape %num_elements, %dynamic_shape
           : (index, tensor<2xi32>) -> tensor<2xi32>
    ```
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs 1DTensorOf<[AnyInteger, Index]>:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_CstrReshapableOp :
    StableHLO_Op<"cstr_reshapable", [Pure]> {
  let summary = "CstrReshapable operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation creates a witness on the constraint that
    ComputeReshapeShape would succeed with the provided operands.

    Example:
    ```mlir
    %result = stablehlo.cstr_reshapable %num_elements, %dynamic_shape
           : (index, tensor<3xi32>) -> !shape.witness
    ```
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs Shape_WitnessType:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // STABLEHLO_DIALECT_STABLEHLO_OPS
