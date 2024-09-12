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
include "mlir/IR/SymbolInterfaces.td"
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

class StableHLO_Op<string mnemonic, list<Trait> traits = []> :
    Op<StableHLO_Dialect, mnemonic, traits> {
  string commonClassDeclaration = [{
    // Relax the strict default implementation with one that allows
    // for StableHLO-specific differences.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
  let extraClassDeclaration = commonClassDeclaration;
}

include "stablehlo/dialect/StablehloEnums.td"
include "stablehlo/dialect/StablehloAttrs.td"
include "stablehlo/dialect/StablehloTypes.td"

class StableHLO_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    StableHLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {}

//===----------------------------------------------------------------------===//
// StableHLO nullary op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_ConstantOp : StableHLO_Op<"constant",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
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
    HLO_StaticShapeTensorOrPerAxisQuantizedTensor:$output
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
  let arguments = (ins
    ConfinedAttr<I64Attr, [IntNonNegative]>:$iota_dimension /*iota_c1*/
  );

  let results = (outs HLO_StaticShapeIntFpComplexOrQuantizedTensor:$output);

  let hasVerifier = 1;

  let assemblyFormat = "`dim` `=` $iota_dimension attr-dict `:` type($output)";
}

def StableHLO_DynamicIotaOp: StableHLO_ShapedInterfaceOp<"dynamic_iota", [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "DynamicIota operation";
  let description = [{
    This operation is functionally identical to
    [iota](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota)
    op, but the result shape is specified dynamically via `output_shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_iota

    Example:
    ```mlir
    %output_shape = stablehlo.constant dense<[4, 5]> : tensor<2xi64>
    %0 = stablehlo.dynamic_iota %output_shape, dim = 0 : (tensor<2xi64>) -> tensor<4x5xi64>
    ```
  }];

  let arguments = (ins
    HLO_StaticDimensionTensor:$output_shape /*dynamic_iota_i1*/,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$iota_dimension /*dynamic_iota_c1, dynamic_iota_i2*/
  );
  let results = (outs HLO_Tensor:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $output_shape `,` `dim` `=` $iota_dimension attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
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
    Type OperandType, Type ResultType = OperandType> : StableHLO_Op<mnemonic,
    traits # [Elementwise, InferShapedTypeOpInterface, SameOperandsAndResultShape,
              HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect]> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
  let extraClassDeclaration = commonClassDeclaration # [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                operands.front(),
                                                &reifiedReturnShapes);
    }
  }];

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

// Abs supports complex to real, so element type is not guaranteed to match.
def StableHLO_AbsOp: StableHLO_UnaryElementwiseOp<"abs",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>],
     RankedTensorOf<[HLO_SInt, HLO_Float, HLO_Complex, HLO_QuantizedInt] /* abs_i1 */>,
     RankedTensorOf<[HLO_SInt, HLO_Float, HLO_QuantizedInt]>> {
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
    [HLO_CompatibleOperandsAndResultType /*cbrt_c1*/],
    HLO_FpComplexOrQuantizedIntTensor /*cbrt_i1*/> { /*cbrt_c1*/
  let summary = "Cbrt operation";
  let description = [{
    Performs element-wise cubic root operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cbrt

    Example:
    ```mlir
    %result = stablehlo.cbrt %operand : tensor<4xf64>
    ```
  }];
}

def StableHLO_CeilOp: StableHLO_UnaryElementwiseOp<"ceil",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrQuantizedIntTensor> {
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
    [SameOperandsAndResultShape /*convert_c1*/], HLO_NonQuantizedTensor> { /*convert_i1*/
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xcomplex<f64>>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$operand, "Type":$result_element_ty)>];
}

def StableHLO_ClzOp: StableHLO_UnaryElementwiseOp<"count_leading_zeros",
    [HLO_CompatibleOperandsAndResultType /*count_leading_zeros_c1*/],
     HLO_IntTensor /*count_leading_zeros_i1*/> { /*count_leading_zeros_c1*/
  let summary = "Clz operation";
  let description = [{
    Performs element-wise count of the number of leading zero bits in the
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros

    Example:
    ```mlir
    %result = stablehlo.count_leading_zeros %operand : tensor<2x2xi64>
    ```
  }];
}

def StableHLO_CosineOp: StableHLO_UnaryElementwiseOp<"cosine",
    [HLO_CompatibleOperandsAndResultType], HLO_FpComplexOrQuantizedIntTensor> {
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
    [HLO_CompatibleOperandsAndResultType /*exponential_c1*/],
     HLO_FpComplexOrQuantizedIntTensor /*exponential_i1*/> {
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
    [HLO_CompatibleOperandsAndResultType], /*exponential_minus_one_c1*/
    HLO_FpComplexOrQuantizedIntTensor /*exponential_minus_one_i1*/> { /*exponential_minus_one_c1*/
  let summary = "Expm1 operation";
  let description = [{
    Performs element-wise exponential minus one operation on `operand` tensor
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one

    Example:
    ```mlir
    %result = stablehlo.exponential_minus_one %operand : tensor<2xf64>
    ```
  }];
}

def StableHLO_FloorOp: StableHLO_UnaryElementwiseOp<"floor",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrQuantizedIntTensor> {
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
    [DeclareOpInterfaceMethods<InferTypeOpInterface>],
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

def StableHLO_IsFiniteOp: StableHLO_UnaryElementwiseOp<"is_finite",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>], HLO_FpOrQuantizedIntTensor> {
    /*is_finite_c1*/
  let summary = "IsFinite operation";
  let description = [{
    Performs element-wise check whether the value in `x` is finite (i.e. is
    neither +Inf, -Inf, nor NaN) and produces a `y` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#is_finite

    Example:
    ```mlir
    %y = stablehlo.is_finite %x : (tensor<7xf64>) -> tensor<7xi1>
    ```
  }];
  let arguments = (ins HLO_FpOrQuantizedIntTensor:$x); /*is_finite_i1*/
  let results = (outs HLO_PredTensor:$y);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_LogOp: StableHLO_UnaryElementwiseOp<"log",
    [HLO_CompatibleOperandsAndResultType /*log_c1*/],
     HLO_FpComplexOrQuantizedIntTensor /*log_i1*/> {
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
    [HLO_CompatibleOperandsAndResultType /*log_plus_one_c1*/],
    HLO_FpComplexOrQuantizedIntTensor /*log_plus_one_i1*/> { /*log_plus_one_c1*/
  let summary = "Log1p operation";
  let description = [{
    Performs element-wise logarithm plus one operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one

    Example:
    ```mlir
    %result = stablehlo.log_plus_one %operand : tensor<5xf64>
    ```
  }];
}

def StableHLO_LogisticOp: StableHLO_UnaryElementwiseOp<"logistic",
    [HLO_CompatibleOperandsAndResultType /*logistic_c1*/],
    HLO_FpComplexOrQuantizedIntTensor /*logistic_i1*/> { /*logistic_c1*/
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
    [HLO_CompatibleOperandsAndResultType], HLO_PredOrIntTensor> {
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
    [HLO_CompatibleOperandsAndResultType], HLO_IntFpOrComplexOrQuantizedIntTensor> {
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
    [HLO_CompatibleOperandsAndResultType /*popcnt_c1*/],
    HLO_IntTensor /*popcnt_i1*/> { /*popcnt_c1*/
  let summary = "PopulationCount operation";
  let description = [{
    Performs element-wise count of the number of bits set in the `operand`
    tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt

    Example:
    ```mlir
    %result = stablehlo.popcnt %operand : tensor<4xi64>
    ```
  }];
}

def StableHLO_RealOp: StableHLO_UnaryElementwiseOp<"real",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>],
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
    [HLO_CompatibleOperandsAndResultType /*round_nearest_afz_c1*/],
    HLO_FpOrQuantizedIntTensor /*round_nearest_afz_i1*/> { /*round_nearest_afz_c1*/
  let summary = "Round operation";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties
    away from zero, on the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_afz

    Example:
    ```mlir
    %result = stablehlo.round_nearest_afz %operand : tensor<5xf64>
    ```
  }];
}

def StableHLO_RoundNearestEvenOp: StableHLO_UnaryElementwiseOp<"round_nearest_even",
    [HLO_CompatibleOperandsAndResultType /*round_nearest_even_c1*/],
    HLO_FpOrQuantizedIntTensor /*round_nearest_even_i1*/> { /*round_nearest_even_c1*/
  let summary = "RoundNearestEven operation";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties
    towards the even integer, on the `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_even

    Example:
    ```mlir
    %result = stablehlo.round_nearest_even %operand : tensor<5xf64>
    ```
  }];
}

def StableHLO_RsqrtOp: StableHLO_UnaryElementwiseOp<"rsqrt",
    [HLO_CompatibleOperandsAndResultType /* rsqrt_c1 */],
    HLO_FpComplexOrQuantizedIntTensor /* rsqrt_i1 */> {
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
    [HLO_CompatibleOperandsAndResultType /*sign_c1*/],
    RankedTensorOf<[HLO_SInt, HLO_Float, HLO_Complex, HLO_QuantizedInt]> /*sign_i1*/> { /*sign_c1*/
  let summary = "Sign operation";
  let description = [{
    Returns the sign of the `operand` element-wise and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    Example:
    ```mlir
    %result = stablehlo.sign %operand : tensor<5xf64>
    ```
  }];
}

def StableHLO_SineOp: StableHLO_UnaryElementwiseOp<"sine",
    [HLO_CompatibleOperandsAndResultType], HLO_FpComplexOrQuantizedIntTensor> {
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

def StableHLO_SqrtOp: StableHLO_UnaryElementwiseOp<"sqrt",
    [HLO_CompatibleOperandsAndResultType /* sqrt_c1 */],
    HLO_FpComplexOrQuantizedIntTensor /* sqrt_i1 */> {
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

def StableHLO_TanOp: StableHLO_UnaryElementwiseOp<"tan",
    [HLO_CompatibleOperandsAndResultType /* tan_c1 */],
    HLO_FpComplexOrQuantizedIntTensor /* tan_i1 */> {
  let summary = "Tan operation";
  let description = [{
    Performs element-wise tangent operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tan

    Example:
    ```mlir
    %result = stablehlo.tan %operand : tensor<2x2xf64>
    ```
  }];
}

def StableHLO_TanhOp: StableHLO_UnaryElementwiseOp<"tanh",
    [HLO_CompatibleOperandsAndResultType /* tanh_c1 */],
    HLO_FpComplexOrQuantizedIntTensor /* tanh_i1 */> {
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
    SameOperandsAndResultShape, Elementwise,
    HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect]> {
  let arguments = (ins
    OperandType:$lhs,
    OperandType:$rhs
  );

  string binaryElementwiseOpCommonClassDeclaration = commonClassDeclaration # [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                 operands.front(),
                                                 &reifiedReturnShapes);
    }
  }];

  let extraClassDeclaration = binaryElementwiseOpCommonClassDeclaration;

  let results = (outs ResultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def StableHLO_AddOp : StableHLO_BinaryElementwiseOp<"add", [HLO_Commutative,
      InferTypeOpInterface,
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>],
      HLO_TensorOrPerAxisQuantizedTensor> {
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

  let extraClassDeclaration = binaryElementwiseOpCommonClassDeclaration # [{
    static LogicalResult inferReturnTypes(
        MLIRContext * /*context*/, std::optional<Location> location,
        ValueRange operands, DictionaryAttr /*attributes*/,
        OpaqueProperties /*properties*/, RegionRange /*regions*/,
        SmallVectorImpl<Type> &inferredReturnTypes) {
      if (operands.empty())
        return emitOptionalError(
            location,
            "Expected non-empty operands for AddOp::inferReturnTypes");

      auto inferredTypeOrErr =
          mlir::hlo::inferMostSpecificType(location, operands.getTypes());
      if (failed(inferredTypeOrErr)) return failure();
      inferredReturnTypes.emplace_back(*inferredTypeOrErr);
      return success();
    }
  }];

  let hasVerifier = 1;

}

def StableHLO_Atan2Op : StableHLO_BinaryElementwiseOp<"atan2",
      [HLO_CompatibleOperandsAndResultType /*atan2_c1*/],
      HLO_FpComplexOrQuantizedIntTensor /*atan2_i1, atan2_i2*/> { /*atan2_c1*/
  let summary = "Atan2 operation";
  let description = [{
    Performs element-wise atan2 operation on `lhs` and `rhs` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#atan2

    Example:
    ```mlir
    %result = stablehlo.atan2 %lhs, %rhs : tensor<3xf64>
    ```
  }];
}

def StableHLO_ComplexOp: StableHLO_BinaryElementwiseOp<"complex",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    SameOperandsElementType /*complex_c1*/,
    SameOperandsAndResultShape /*complex_c2*/,
    DeclareOpInterfaceMethods<InferTypeOpInterface> /*complex_c3*/]> {
  let summary = "Complex operation";
  let description = [{
    Performs element-wise conversion to a complex value from a pair of real and
    imaginary values, `lhs` and `rhs`, and produces a `result` tensor.
    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#complex
    Example:
    ```mlir
    %result = stablehlo.complex %lhs, %rhs : tensor<2xcomplex<f64>>
    ```
  }];
  let arguments = (ins
     HLO_Fp32Or64Tensor:$lhs /*complex_i1*/,
     HLO_Fp32Or64Tensor:$rhs /*complex_i2*/
  );
  let results = (outs
     HLO_ComplexTensor:$result
  );

  let assemblyFormat = [{
    operands attr-dict
      `:` custom<ComplexOpType>(type($lhs), type($rhs), type($result))
  }];
}

def StableHLO_DivOp : StableHLO_BinaryElementwiseOp<"divide",
    [HLO_CompatibleOperandsAndResultType /* div_c1 */],
    HLO_IntFpOrComplexOrQuantizedIntTensor /* div_i1, div_i2 */> {
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
      [HLO_Commutative, HLO_CompatibleOperandsAndResultType]> {
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
      [HLO_Commutative, HLO_CompatibleOperandsAndResultType]> {
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
      [HLO_Commutative, HLO_CompatibleOperandsAndResultType]> {
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
      [HLO_CompatibleOperandsAndResultType /* pow_c1 */],
      HLO_IntFpOrComplexOrQuantizedIntTensor /* pow_i1, pow_i2 */> {
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
      [HLO_CompatibleOperandsAndResultType /*remainder_c1*/],
      HLO_IntFpOrComplexOrQuantizedIntTensor /*remainder_i1, remainder_i2*/> {
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
      [HLO_CompatibleOperandsAndResultType /*shift_left_c1*/],
      HLO_IntTensor /*shift_left_i1, shift_left_i2*/> { /*shift_left_c1*/
  let summary = "ShiftLeft operation";
  let description = [{
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs`
    number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left

    Example:
    ```mlir
    %result = stablehlo.shift_left %lhs, %rhs : tensor<3xi64>
    ```
  }];
}

def StableHLO_ShiftRightArithmeticOp : StableHLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [HLO_CompatibleOperandsAndResultType /*shift_right_arithmetic_c1*/],
      HLO_IntTensor /*shift_right_arithmetic_i1, shift_right_arithmetic_i2*/> { /*shift_right_arithmetic_c1*/
  let summary = "ShiftRightArithmetic operation";
  let description = [{
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor
    by `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic

    Example:
    ```mlir
    %result = stablehlo.shift_right_arithmetic %lhs, %rhs : tensor<3xi64>
    ```
  }];
}

def StableHLO_ShiftRightLogicalOp : StableHLO_BinaryElementwiseOp<"shift_right_logical",
      [HLO_CompatibleOperandsAndResultType /*shift_right_logical_c1*/],
      HLO_IntTensor /*shift_right_logical_i1, shift_right_logical_i2*/> { /*shift_right_logical_c1*/
  let summary = "ShiftRightLogical operation";
  let description = [{
    Performs element-wise logical right-shift operation on the `lhs` tensor by
    `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical

    Example:
    ```mlir
    %result = stablehlo.shift_right_logical %lhs, %rhs : tensor<3xi64>
    ```
  }];
}

def StableHLO_SubtractOp : StableHLO_BinaryElementwiseOp<"subtract",
      [HLO_CompatibleOperandsAndResultType], HLO_IntFpOrComplexOrQuantizedIntTensor> {
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
          [HLO_Commutative, HLO_CompatibleOperandsAndResultType,
           HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect]> {
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

def StableHLO_InfeedOp : StableHLO_Op<"infeed"> {
  let summary = "Infeed operation";
  let description = [{
    Reads data from the infeed and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#infeed

    Example:
    ```mlir
    %results0:2 = "stablehlo.infeed"(%token) :
        (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    ```
  }];

  let arguments = (ins
    HLO_Token:$token, /*infeed_i1*/
    DefaultValuedStrAttr<StrAttr, "">:$infeed_config, /*infeed_i2*/
    OptionalAttr<ArrayAttr>:$layout
  );
  let results = (outs Variadic<HLO_StaticShapeTensorOrPerAxisQuantizedTensorOrToken>);
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
    %result = "stablehlo.outfeed"(%input0, %token) :
        (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
    ```
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrPerAxisQuantizedTensor>:$inputs, /*outfeed_i1*/
    HLO_Token:$token, /*outfeed_i2*/
    DefaultValuedStrAttr<StrAttr, "">:$outfeed_config /*outfeed_i3*/
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
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
      is_host_transfer = true
    } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
    ```
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrPerAxisQuantizedTensor>:$inputs, /*send_i1*/
    HLO_Token:$token, /*send_i2*/
    StableHLO_ChannelHandle:$channel_handle, /*send_i3_i4*/
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer /*send_i5*/
  );

  let results = (outs HLO_Token);
}

def StableHLO_RecvOp : StableHLO_Op<"recv"> {
  let summary = "Recv operation";
  let description = [{
    Receives data from a channel with `channel_id` and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#recv

    Example:
    ```mlir
    %results:2 = "stablehlo.recv"(%token) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
      is_host_transfer = true
    } : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    ```
  }];

  let arguments = (ins
    HLO_Token:$token, /*recv_i1*/
    StableHLO_ChannelHandle:$channel_handle, /*recv_i2_i3*/
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer /*recv_i4*/
  );

  let results = (outs Variadic<HLO_StaticShapeTensorOrPerAxisQuantizedTensorOrToken>);
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
  let results = (outs UI32RankedTensor);

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
  let results = (outs UI32RankedTensor);

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

  let arguments = (ins Variadic<HLO_Token>:$inputs /*after_all_i1*/);
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
    RecursivelySpeculatable,
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

  let results = (outs Variadic<HLO_TensorOrPerAxisQuantizedTensorOrToken>);
}

// Xla Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. CaseOp maps to indexed
// conditional use of kConditional HLO.
def StableHLO_CaseOp: StableHLO_Op<"case", [
      RecursiveMemoryEffects,
      RecursivelySpeculatable,
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
    I32RankedTensor:$index /*case_i1*/
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches /*case_i2*/);

  let results = (outs Variadic<HLO_TensorOrPerAxisQuantizedTensorOrToken>);
}


def StableHLO_WhileOp: StableHLO_Op<"while", [
      RecursiveMemoryEffects,
      RecursivelySpeculatable,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*while_c3*/,
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
    %results0, %results1 = stablehlo.while(%arg0 = %init_i, %arg1 = %init_sum) : tensor<i64>, tensor<i64>
    cond {
      %cond = stablehlo.compare LT, %arg0, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %new_sum = stablehlo.add %arg1, %one : tensor<i64>
      %new_i = stablehlo.add %arg0, %one : tensor<i64>
      stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }
    ```
  }];
  let arguments = (ins Variadic<HLO_TensorOrPerAxisQuantizedTensorOrToken>:$operand /*while_i1*/);

  let regions = (region
    SizedRegion<1>:$cond /*while_i2*/,
    SizedRegion<1>:$body /*while_i3*/
  );

  let results = (outs Variadic<HLO_TensorOrPerAxisQuantizedTensorOrToken>);

  let hasVerifier = 1;

  let extraClassDeclaration = commonClassDeclaration # [{
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
  }];
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_AllGatherOp : StableHLO_Op<"all_gather",
    [ConditionallySpeculatable,
     HLO_PairwiseSameOperandAndResultElementType, /*all_gather_c6*/]> {
  string summary = "AllGather operation";
  string description = [{
    Within each process group in the process grid, concatenates the values of the
    `operand` tensor from each process along `all_gather_dim` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_gather

    Example:
    ```mlir
    %result:2 = "stablehlo.all_gather"(%operand0, %operand1) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$operands, /*all_gather_i1*/
    ConfinedAttr<I64Attr, [IntNonNegative]>:$all_gather_dim, /*all_gather_c1, all_gather_i2*/
    I64ElementsAttr:$replica_groups, /*all_gather_i3*/
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle, /*all_gather_i4*/
    UnitAttr:$use_global_device_ids /*all_gather_i5*/
  );
  let results = (outs Variadic<HLO_Tensor>);
  let hasVerifier = 1;

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_AllReduceOp : StableHLO_Op<"all_reduce",
    [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput,
     InferTensorType /*all_reduce_c6, all_reduce_c7*/,
     SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "AllReduce operation";
  let description = [{
    Within each process group in the process grid, applies a reduction function
    `computation` to the values of the `operand` tensor from each process and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_reduce

    Example:
    ```mlir
    %result:2 = "stablehlo.all_reduce"(%operand0, %operand0) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%0) : (tensor<i64>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$operands, /*all_reduce_i1*/
    I64ElementsAttr:$replica_groups, /*all_reduce_i2*/
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle, /*all_reduce_i3*/
    UnitAttr:$use_global_device_ids /*all_reduce_i4*/
  );
  let regions = (region SizedRegion<1>:$computation /*all_reduce_i5*/);
  let results = (outs Variadic<HLO_Tensor>);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::DenseIntElementsAttr":$replica_groups,
      "::mlir::stablehlo::ChannelHandleAttr":$channel_handle,
      CArg<"bool", "false">:$use_global_device_ids)>,
 ];
}

def StableHLO_ReduceScatterOp : StableHLO_Op<"reduce_scatter", [ConditionallySpeculatable]> {
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
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<2x4xi64>) -> tensor<2x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand, /*reduce_scatter_i1*/
    ConfinedAttr<I64Attr, [IntNonNegative]>:$scatter_dimension, /*reduce_scatter_c2, reduce_scatter_i2*/
    I64ElementsAttr:$replica_groups, /*reduce_scatter_i3*/
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle, /*reduce_scatter_i4*/
    UnitAttr:$use_global_device_ids /*reduce_scatter_i5*/
  );
  let regions = (region SizedRegion<1>:$computation /*reduce_scatter_i6*/);
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_AllToAllOp : StableHLO_Op<"all_to_all",
    [ConditionallySpeculatable,
     HLO_PairwiseSameOperandAndResultElementType /*all_to_all_c9*/,
     InferTensorType /*all_to_all_c9*/]> {
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
    %result:2 = "stablehlo.all_to_all"(%operand1, %operand2) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$operands, /*all_to_all_i1*/
    ConfinedAttr<I64Attr, [IntNonNegative]>:$split_dimension, /*all_to_all_c1, all_to_all_i2*/
    ConfinedAttr<I64Attr, [IntNonNegative]>:$concat_dimension, /*all_to_all_c3, all_to_all_i3*/
    ConfinedAttr<I64Attr, [IntPositive]>:$split_count, /*all_to_all_c4, all_to_all_i4*/
    I64ElementsAttr:$replica_groups, /*all_to_all_i5*/
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle /*all_to_all_i6*/
  );
  let results = (outs Variadic<HLO_Tensor>);

  // channel_handle is only used for the SPMD partitioner, so we add a
  // simplified builder method for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::IntegerAttr": $split_dimension,
      "::mlir::IntegerAttr": $concat_dimension,
      "::mlir::IntegerAttr": $split_count,
      "::mlir::DenseIntElementsAttr": $replica_groups)>];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_ReduceOp: StableHLO_ShapedInterfaceOp<"reduce", [
      HLO_RecursivelySpeculatableIfAllInputsStatic,
      RecursiveMemoryEffects,
      SameVariadicOperandSize /*reduce_c3*/,
      InferTensorTypeWithReify /*reduce_c7, reduce_c8*/,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]> { /*reduce_c7*/
  let summary = "Reduce operation";
  let description = [{
    Applies a reduction function `body` to `inputs` and `init_values` along the
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs, /*reduce_i1*/
    Variadic<HLO_Tensor>:$init_values, /*reduce_i2*/
    DenseI64ArrayAttr:$dimensions /*reduce_i3*/
  );
  let regions = (region SizedRegion<1>:$body /*reduce_i4*/);

  // Builder
  // The following custom builder allows inferring the operation type using the
  // 'element_types' of the arguments of the 'body'.
  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$init_values,
      "DenseI64ArrayAttr":$dimensions, "TypeRange":$element_types)>,
  ];

  let results = (outs Variadic<HLO_Tensor>);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

}

//===----------------------------------------------------------------------===//
// StableHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def StableHLO_GetTupleElementOp: StableHLO_Op<"get_tuple_element", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface> /*get_tuple_element_c2*/]> {
  let summary = "GetTupleElement operation";
  let description = [{
    Extracts element at `index` position of the `operand` tuple and produces a
    `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_tuple_element

    Example:
    ```mlir
    %result = stablehlo.get_tuple_element %operand[0] : (tuple<tensor<2xf64>, tuple<tensor<i64>>>) -> tensor<2xf64>
    ```
  }];
  let arguments = (ins
    HLO_Tuple:$operand, /*get_tuple_element_i1*/
    ConfinedAttr<I32Attr, [IntNonNegative]>:$index /*get_tuple_element_c1, get_tuple_element_i2*/
  );

  let results = (outs HLO_TensorOrPerAxisQuantizedTensorOrTokenOrTuple);

  let assemblyFormat = [{
    $operand `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_TupleOp : StableHLO_Op<"tuple", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface> /*tuple_c1*/]> {
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from values `val`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tuple

    Example:
    ```mlir
    %result = stablehlo.tuple %val0, %val1 : tuple<tensor<2xf64>, tuple<tensor<i64>>>
    ```
   }];

  let arguments = (ins Variadic<HLO_TensorOrPerAxisQuantizedTensorOrTokenOrTuple>:$val /*tuple_i1*/);
  let results = (outs HLO_Tuple:$result);

  let assemblyFormat = [{
    $val attr-dict `:` custom<TupleOpType>(type($val), type($result))
  }];
}

def StableHLO_CompareOp: StableHLO_Op<"compare",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    Elementwise,
    HLO_CompatibleOperandsElementType /*compare_c1*/,
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
      [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect,
      SameOperandsAndResultElementType /*slice_c1*/,
      AllMatchSameOperatorTrait<["start_indices", "limit_indices",
      "strides"], "$_self.size()", "size"> /*slice_c2*/,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from the `operand` using statically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = stablehlo.slice %operand [1:3, 4:8:2]
       : (tensor<3x8xi64>) -> tensor<2x2xi64>

    // Same in generic form: the `1:3` above is mapped to the first entry in
    // `start_indices` and `limit_indices`, while `strides` is implicitly 1.
    // The `4:8:2` above is parsed into the second entry of `start_indices`,
    // `limit_indices` and `strides` respectively.
    %result = "stablehlo.slice" (%operand) {
      start_indices = array<i64: 1, 4>,
      limit_indices = array<i64: 3, 8>,
      strides = array<i64: 1, 2>
    } : (tensor<3x8xi64>) -> tensor<2x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    DenseI64ArrayAttr:$start_indices,
    DenseI64ArrayAttr:$limit_indices,
    DenseI64ArrayAttr:$strides
  );

  let assemblyFormat = [{
    $operand custom<SliceRanges>($start_indices, $limit_indices, $strides)
      attr-dict `:` functional-type(operands, results)
  }];

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
    DenseI64ArrayAttr:$slice_sizes /*dynamic_slice_i3*/
  );

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $operand `,` custom<VariadicOperandWithAttribute>($start_indices)
      `sizes` `=` $slice_sizes attr-dict `:` functional-type(operands, results)
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

def StableHLO_BatchNormGradOp : StableHLO_Op<"batch_norm_grad",
   [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    HLO_CompatibleOperandsAndResultElementType /*batch_norm_grad_c2*/,
    InferTensorType /*batch_norm_grad_c3, batch_norm_grad_c4*/]> {
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
    } : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>,
         tensor<2x2x2xf64>) -> (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
    ```
  }];

  let arguments = (ins
    RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>:$operand, /*batch_norm_grad_i1*/
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$scale, /*batch_norm_grad_i2*/
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$mean, /*batch_norm_grad_i3*/
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$variance, /*batch_norm_grad_i4*/
    RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>:$grad_output, /*batch_norm_grad_i5*/
    F32Attr:$epsilon, /*batch_norm_grad_i6*/
    ConfinedAttr<I64Attr, [IntNonNegative]>:$feature_index /*batch_norm_grad_c1, batch_norm_grad_i7*/
  );

  let results = (outs
      RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>:$grad_operand,
      1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$grad_scale,
      1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$grad_offset);
}

def StableHLO_BatchNormInferenceOp : StableHLO_Op<"batch_norm_inference",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
     HLO_CompatibleOperandsAndResultElementType /*batch_norm_inference_c2*/,
     InferTensorType /*batch_norm_inference_c7*/]> {
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
    } : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<2x2x2xf64>
    ```
  }];

  let arguments = (ins
    RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>:$operand /*batch_norm_inference_i1*/,
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$scale /*batch_norm_inference_i2*/,
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$offset /*batch_norm_inference_i3*/,
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$mean /*batch_norm_inference_i4*/,
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$variance /*batch_norm_inference_i5*/,
    F32Attr:$epsilon /*batch_norm_inference_i6*/,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$feature_index /*batch_norm_inference_c1, batch_norm_inference_i7*/
  );

  let results = (outs RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>:$result);
}

def StableHLO_BatchNormTrainingOp : StableHLO_Op<"batch_norm_training",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
     HLO_CompatibleOperandsAndResultElementType /*batch_norm_training_c2*/,
     InferTensorType /*batch_norm_training_c5, batch_norm_training_c6, batch_norm_training_c7*/]> {
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
    } : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>) ->
        (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
    ```
  }];

  let arguments = (ins
    RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>:$operand /*batch_norm_training_i1*/,
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$scale /*batch_norm_training_i2*/,
    1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$offset /*batch_norm_training_i3*/,
    F32Attr:$epsilon /*batch_norm_training_i4*/,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$feature_index /*batch_norm_training_c1, batch_norm_training_i5*/
  );

  let results = (outs
      RankedTensorOf<[HLO_Float, HLO_QuantizedInt]>:$output,
      1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$batch_mean,
      1DTensorOf<[HLO_Float, HLO_QuantizedInt]>:$batch_var);
}

def StableHLO_BitcastConvertOp : StableHLO_ShapedInterfaceOp<"bitcast_convert",
    [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "BitcastConvert operation";
  let description = [{
    Performs a bitcast operation on `operand` tensor and produces a `result`
    tensor where the bits of the entire `operand` tensor are reinterpreted using
    the type of the `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert

    Example:
    ```mlir
    %result = stablehlo.bitcast_convert %operand : (tensor<f64>) -> tensor<4xf16>
    ```
  }];

  let arguments = (ins HLO_TensorOrPerAxisQuantizedTensor:$operand /*bitcast_convert_i1*/);
  let results = (outs HLO_TensorOrPerAxisQuantizedTensor);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
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
    DenseI64ArrayAttr:$broadcast_sizes
  );

  let results = (outs HLO_TensorOrPerAxisQuantizedTensor);

  let assemblyFormat = [{
    $operand `,` `sizes` `=` $broadcast_sizes
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_BroadcastInDimOp : StableHLO_Op<"broadcast_in_dim",
      [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
      HLO_CompatibleOperandsAndResultElementType /*broadcast_in_dim_c1*/]> {
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
    HLO_TensorOrPerAxisQuantizedTensor:$operand /*broadcast_in_dim_i1*/,
    DenseI64ArrayAttr:$broadcast_dimensions /*broadcast_in_dim_i2*/
  );

  let results = (outs HLO_StaticShapeTensorOrPerAxisQuantizedTensor);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` `dims` `=` $broadcast_dimensions
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Determines if this BroadcastInDim instance can be expressed as a Broadcast.
    bool isSimpleBroadcast();
  }];
}

def StableHLO_DynamicBroadcastInDimOp : StableHLO_ShapedInterfaceOp<
    "dynamic_broadcast_in_dim",
    [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "DynamicBroadcastInDim operation";
  let description = [{
    This operation is functionally identical to
    [broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim)
    op, but the result shape is specified dynamically via `output_dimensions`.

    It also accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_broadcast_in_dim

    Example:
    ```mlir
    %operand = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
    %output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
    %result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
      broadcast_dimensions = array<i64: 2, 1>,
      known_expanding_dimensions = array<i64: 0>,
      known_nonexpanding_dimensions = array<i64: 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    ```

  }];

  let arguments = (ins
    HLO_TensorOrPerAxisQuantizedTensor:$operand /*dynamic_broadcast_in_dim_i1*/,
    HLO_StaticDimensionTensor:$output_dimensions /*dynamic_broadcast_in_dim_i2*/,
    DenseI64ArrayAttr:$broadcast_dimensions /*dynamic_broadcast_in_dim_i3*/,
    OptionalAttr<DenseI64ArrayAttr>:$known_expanding_dimensions /*dynamic_broadcast_in_dim_i4*/,
    OptionalAttr<DenseI64ArrayAttr>:$known_nonexpanding_dimensions /*dynamic_broadcast_in_dim_i5*/
  );

  let results = (outs HLO_TensorOrPerAxisQuantizedTensor);

  let builders = [
      OpBuilder<(ins
        "Type":$result_type, "Value":$operand, "Value":$output_dimensions,
        "DenseI64ArrayAttr":$broadcast_dimensions), [{
      build($_builder, $_state, result_type, operand, output_dimensions,
          broadcast_dimensions, /*known_expanding_dimensions=*/{},
          /*known_nonexpanding_dimensions=*/{});
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` $output_dimensions `,` `dims` `=` $broadcast_dimensions
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

// Note: There is no HLO_CallOp because the standard call operation mlir::func::CallOp
// is used instead. A mlir::func::CallOp is exported to a HLO call instruction
// directly.

def StableHLO_CholeskyOp : StableHLO_Op<"cholesky",
      [NoMemoryEffect,
       HLO_CompatibleOperandsAndResultElementType /*cholesky_c1*/,
       InferTensorType /*cholesky_c1*/]> {
  let summary = "Cholesky operation";
  let description = [{
    Computes the Cholesky decomposition of a batch of matrices.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cholesky

    Example:
    ```mlir
    %result = stablehlo.cholesky %a, lower = true : tensor<3x3xf64>
    ```
  }];
  let arguments = (ins
    HLO_FpComplexOrQuantizedIntTensor:$a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$lower
  );

  let results = (outs HLO_FpComplexOrQuantizedIntTensor:$result);

  let assemblyFormat = [{
    $a (`,` `lower` `=` $lower^)? attr-dict `:` custom<SameOperandsAndResultType>(type($a), type($result))
  }];
}

def StableHLO_ClampOp : StableHLO_ShapedInterfaceOp<"clamp",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    HLO_CompatibleOperandsAndResultElementType  /* clamp_c3 */, HLO_BroadcastingElementwise,
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
    [ConditionallySpeculatable, NoMemoryEffect,
     SameOperandsAndResultElementType /*concatenate_c1, concatenate_c3, concatenate_c5*/,
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
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension /*concatenate_c4, concatenate_i2*/
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
     custom<VariadicOperandWithAttribute>($inputs) `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}


def StableHLO_CollectiveBroadcastOp: StableHLO_Op<"collective_broadcast",
    [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput,
     SameOperandsAndResultElementType /*collective_broadcast_c3*/,
     DeclareOpInterfaceMethods<InferTypeOpInterface>,
     DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>
    ]> {
  let summary = "CollectiveBroadcast operation";
  let description = [{
    Within each process group in the process grid, send the value of the
    `operand` tensor from the source process to the target processes and produce a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective_broadcast

    Example:
    ```mlir
    %result = "stablehlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand, /*collective_broadcast_i1*/
    I64ElementsAttr:$replica_groups, /*collective_broadcast_i2*/
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle /*collective_broadcast_i3*/
  );
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;
  // channel_handle is only used for the SPMD partitioner, so we add a
  // simplified builder method for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::DenseIntElementsAttr":$replica_groups)>];
}

def StableHLO_CollectivePermuteOp: StableHLO_Op<"collective_permute",
    [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput,
     SameOperandsAndResultElementType, /*collective_permute_c5*/
     DeclareOpInterfaceMethods<InferTypeOpInterface>,
     DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>
    ]> {
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
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<2x2xi64>) -> tensor<2x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand, /*collective_permute_i1*/
    I64ElementsAttr:$source_target_pairs, /*collective_permute_i2*/
    OptionalAttr<StableHLO_ChannelHandle>:$channel_handle /*collective_permute_i3*/
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

def StableHLO_CompositeOp : StableHLO_Op<"composite", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Composite operation";
  let description = [{
    Encapsulates an operation made up (composed) of other StableHLO operations,
    taking `inputs` and `composite_attributes` and producing `results`. The
    semantics of the op are implemented by the `decomposition` attribute. The
    `composite` op can be replaced with its decomposition without changing program
    semantics. In cases where inlining the decomposition does not provide the same
    op semantics, prefer using `custom_call`.

    The `version` field (defaults to `0`) is used to denote when a composite's
    semantics change.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#composite

    Example:
    ```mlir
    %results = stablehlo.composite "my.op" %input0, %input1 {
      composite_attributes = {
        my_attribute = "my_value"
      },
      decomposition = @my_op,
      version = 1 : i32
    } : (tensor<f32>, tensor<f32>) -> tensor<f32>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrPerAxisQuantizedTensorOrTokenOrTuple>:$inputs, /*composite_i1*/
    StrAttr:$name, /*composite_i2*/
    DefaultValuedOptionalAttr<DictionaryAttr, "{}">:$composite_attributes, /*composite_i3*/
    FlatSymbolRefAttr:$decomposition, /*composite_i4*/
    DefaultValuedOptionalAttr<I32Attr, "0">:$version /*composite_i5*/
  );
  let results = (outs Variadic<HLO_TensorOrPerAxisQuantizedTensorOrTokenOrTuple>);

  let assemblyFormat = "$name $inputs attr-dict `:` functional-type(operands, results)";
}

def StableHLO_ConvolutionOp : StableHLO_Op<"convolution",
    [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "Convolution operation";
  let description = [{
    Computes dot products between windows of `lhs` and slices of `rhs` and
    produces `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution

    Example:
    ```mlir
    %result = stablehlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {
        stride = [4, 4],
        pad = [[0, 0], [0, 0]],
        lhs_dilate = [2, 2],
        rhs_dilate = [1, 1],
        reverse = [0, 0]
      } {
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64,
        precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
      } :
    (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$lhs, /*convolution_i1*/
    HLO_TensorOrPerAxisQuantizedTensor:$rhs, /*convolution_i2*/
    // Default value: one for each of the spatial dimension.
    OptionalAttr<DenseI64ArrayAttr>:$window_strides, /*convolution_i3*/
    // Default value: two zeros for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding, /*convolution_i4*/
    // Default value: one for each of the spatial dimension.
    OptionalAttr<DenseI64ArrayAttr>:$lhs_dilation, /*convolution_i5*/
    // Default value: one for each of the spatial dimension.
    OptionalAttr<DenseI64ArrayAttr>:$rhs_dilation, /*convolution_i6*/
    // Default value: false for each of the spatial dimension.
    OptionalAttr<DenseBoolArrayAttr>:$window_reversal, /*convolution_i7*/
    StableHLO_ConvDimensionNumbers:$dimension_numbers, /*convolution_i8...convolution_i16*/
    ConfinedAttr<I64Attr, [IntPositive]>:$feature_group_count, /*convolution_c21, convolution_i17*/
    ConfinedAttr<I64Attr, [IntPositive]>:$batch_group_count, /*convolution_c22, convolution_i18*/
    OptionalAttr<StableHLO_PrecisionConfigAttr>:$precision_config /*convolution_i19*/
  );

  let results = (outs HLO_TensorOrPerAxisQuantizedTensor);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasWindowReversal() {
      auto reversal = getWindowReversal();
      return reversal.has_value() && llvm::any_of(reversal.value(), [](bool v) { return v; });
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

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
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

    Depending on the API version there are two ways to pass extra bits of static
    information to the external function:
    1. Use `API_VERSION_TYPED_FFI` which allows passing a dictionary attribute.
    2. Use a previous API version with a StringAttr to encode backend config.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call

    Example:
    ```mlir
    %results = stablehlo.custom_call @foo(%input0) {
      backend_config = {bar = 42 : i32},
      api_version = 4 : i32,
      called_computations = [@foo]
    } : (tensor<f64>) -> tensor<f64>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_CustomCallValue>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    OptionalAttr<AnyAttrOf<[StrAttr, DictionaryAttr]>>:$backend_config,
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

  let results = (outs Variadic<HLO_CustomCallValue>);
  let hasVerifier = 1;

  let assemblyFormat = [{
    custom<CustomCallTarget>($call_target_name) `(` $inputs `)`
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    mlir::Attribute getBackendConfigOrDefault();
    bool hasEmptyBackendConfig();
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
    HLO_TensorOrPerAxisQuantizedTensor:$rhs,
    OptionalAttr<StableHLO_PrecisionConfigAttr>:$precision_config
  );
  let results = (outs HLO_TensorOrPerAxisQuantizedTensor);
  let hasVerifier = 1;

  // Use empty `` to prevent extra whitespace before precision config.
  let assemblyFormat = [{
    $lhs `,` $rhs `` custom<PrecisionConfig>($precision_config) attr-dict
      `:` functional-type(operands, results)
  }];
}

def StableHLO_DotGeneralOp: StableHLO_ShapedInterfaceOp<"dot_general",
    [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "DotGeneral operation";
  let description = [{
    Computes dot products between slices of `lhs` and slices of `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general

    Example:
    ```mlir
    %result = stablehlo.dot_general %lhs, %rhs,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1],
      precision = [DEFAULT, DEFAULT],
      algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
      : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$lhs /*dot_general_i1*/,
    HLO_TensorOrPerAxisQuantizedTensor:$rhs /*dot_general_i2*/,
    StableHLO_DotDimensionNumbers:$dot_dimension_numbers /*dot_general_i3, dot_general_i4, dot_general_i5, dot_general_i6*/,
    OptionalAttr<StableHLO_PrecisionConfigAttr>:$precision_config /*dot_general_i7*/,
    OptionalAttr<StableHLO_DotAlgorithm>:$algorithm
  );

  let results = (outs HLO_TensorOrPerAxisQuantizedTensor);
  let hasVerifier = 1;

  // Use empty `` to prevent extra whitespace before precision config.
  let assemblyFormat = [{
    $lhs `,` $rhs `,` custom<DotDimensionNumbers>($dot_dimension_numbers) ``
    custom<PrecisionConfigAndAlgorithm>($precision_config, $algorithm) attr-dict
      `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();

    /// Determines if this DotGeneral instance can be expressed as a DotOp.
    bool isSimpleDot();
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

def StableHLO_FftOp: StableHLO_Op<"fft",
    [ConditionallySpeculatable, NoMemoryEffect,
    InferTensorType]> {
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
    DenseI64ArrayAttr:$fft_length
  );

  let results = (outs HLO_FpOrComplexTensor);

  let assemblyFormat = [{
    $operand `,` `type` `=` $fft_type `,` `length` `=` $fft_length
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_GatherOp: StableHLO_Op<"gather",
    [ConditionallySpeculatable, NoMemoryEffect,
    InferTensorTypeWithReify /*gather_c22*/,
    AllElementTypesMatch<["operand", "result"]> /*gather_c23*/]> {
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
        offset_dims = [3, 4],
        collapsed_slice_dims = [1],
        operand_batching_dims = [0],
        start_indices_batching_dims = [1],
        start_index_map = [2, 1],
        index_vector_dim = 3>,
      slice_sizes = array<i64: 1, 1, 2, 2>,
      indices_are_sorted = false
    } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand /*gather_i1*/,
    HLO_IntTensor:$start_indices /*gather_i2*/,
    StableHLO_GatherDimensionNumbers:$dimension_numbers /*gather_i3...gather_i8*/,
    DenseI64ArrayAttr:$slice_sizes /*gather_i9*/,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted /*gather_i10*/
  );

  let results = (outs HLO_Tensor:$result);

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_GetDimensionSizeOp: StableHLO_Op<"get_dimension_size",
      [Pure, InferTensorType]> {
  let summary = "GetDimensionSize operation";
  let description = [{
    Produces the size of the given `dimension` of the `operand`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_dimension_size

    Example:
    ```mlir
    %result = stablehlo.get_dimension_size %operand, dim = 1 : (tensor<2x3xi64>) -> tensor<i32>
    ```
  }];
  let arguments = (ins
    HLO_TensorOrPerAxisQuantizedTensor:$operand, /*get_dimension_size_i1*/
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension /*get_dimension_size_c1, get_dimension_size_i2*/
  );
  // TODO(hinsu): Allow 64-bit result types once XLA HLO dialect based on the
  // XLA semantics is available. This limitation is because of the current XLA
  // implementation.
  let results = (outs I32RankedTensor);

  let assemblyFormat = [{
    $operand `,` `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_MapOp: StableHLO_ShapedInterfaceOp<"map",
      [HLO_RecursivelySpeculatableIfAllInputsStatic, RecursiveMemoryEffects,
       SameOperandsAndResultShape /*map_c1, map_c2*/,
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
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 0, 1>
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
    ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*map_i1*/,
    DenseI64ArrayAttr:$dimensions /*map_i2*/
  );
  let regions = (region SizedRegion<1>:$computation /*map_i3*/);
  let results = (outs HLO_Tensor);

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_ReshapeOp: StableHLO_Op<"reshape",
      [ConditionallySpeculatable, NoMemoryEffect,
       HLO_CompatibleOperandsAndResultElementType]> {
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

  let arguments = (ins HLO_TensorOrPerAxisQuantizedTensor:$operand);

  let results = (outs HLO_StaticShapeTensorOrPerAxisQuantizedTensor);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_DynamicReshapeOp: StableHLO_ShapedInterfaceOp<"dynamic_reshape",
      [HLO_SpeculatableIfAllInputsStaticAndShapeConstant, NoMemoryEffect]> {
  let summary = "DynamicReshape operation";
  let description = [{
    This operation is functionally identical to
    [reshape](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape)
    op, but the result shape is specified dynamically via `output_shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_reshape

    Example:
    ```mlir
    %output_shape = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %result = stablehlo.dynamic_reshape %operand, %output_shape : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<3x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_TensorOrPerAxisQuantizedTensor:$operand /*dynamic_reshape_i1*/,
    HLO_StaticDimensionTensor:$output_shape /*dynamic_reshape_i2*/
  );
  let results = (outs HLO_TensorOrPerAxisQuantizedTensor:$result);

  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_ScatterOp: StableHLO_Op<"scatter",
      [ConditionallySpeculatable, RecursiveMemoryEffects,
      SameVariadicOperandSize /*scatter_c5*/,
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*scatter_c24,
      scater_c25*/]> {
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
     ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
       %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
       stablehlo.return %0 : tensor<i64>
   }) {
     scatter_dimension_numbers = #stablehlo.scatter<
       update_window_dims = [3, 4],
       inserted_window_dims = [1],
       input_batching_dims = [0],
       scatter_indices_batching_dims = [1],
       scatter_dims_to_operand_dims = [2, 1],
       index_vector_dim = 3>,
     indices_are_sorted = false,
     unique_indices = false
   } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
   ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs, /*scatter_i1*/
    RankedTensorOf<[AnyInteger, Index]>:$scatter_indices, /*scatter_i2*/
    Variadic<HLO_Tensor>:$updates, /*scatter_i3*/
    StableHLO_ScatterDimensionNumbers:$scatter_dimension_numbers, /*scatter_i4...scatter_i9*/
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted, /*scatter_i10*/
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unique_indices /*scatter_i11*/
  );

  let regions = (region SizedRegion<1>:$update_computation /*scatter_i12*/);

  let results = (outs Variadic<HLO_Tensor>);

  let hasVerifier = 1;

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_SelectOp: StableHLO_Op<"select",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    HLO_BroadcastingElementwise,
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
      [HLO_RecursivelySpeculatableIfStaticDimInOutputIsStaticInInput,
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*select_and_scatter_c11,
      select_and_scatter_c12*/, RecursiveMemoryEffects]> {
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
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %0 : tensor<i1>
    }, {
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      window_dimensions = dense<[3, 1]> : tensor<2xi64>,
      window_strides = dense<[2, 1]> : tensor<2xi64>,
      padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<4x2xi64>, tensor<2x2xi64>, tensor<i64>) -> tensor<4x2xi64>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand, /*select_and_scatter_i1*/
    HLO_Tensor:$source, /*select_and_scatter_i2*/
    HLO_Tensor:$init_value, /*select_and_scatter_i3*/
    OptionalAttr<DenseI64ArrayAttr>:$window_dimensions, /*select_and_scatter_i4*/
    OptionalAttr<DenseI64ArrayAttr>:$window_strides, /*select_and_scatter_i5*/
    OptionalAttr<I64ElementsAttr>:$padding /*select_and_scatter_i6*/
  );

  let regions = (region
    SizedRegion<1>:$select, /*select_and_scatter_i7*/
    SizedRegion<1>:$scatter /*select_and_scatter_i8*/
  );

  let results = (outs HLO_Tensor);

  let hasVerifier = 1;
}

def StableHLO_SetDimensionSizeOp: StableHLO_Op<"set_dimension_size",
      [ConditionallySpeculatable, NoMemoryEffect,
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
    I32RankedTensor:$size,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension
  );
  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` $size  `,` `dim` `=` $dimension attr-dict
      `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_SortOp : StableHLO_Op<"sort",
      [HLO_RecursivelySpeculatableIfAllInputsStatic, RecursiveMemoryEffects,
      SameOperandsAndResultShape /*sort_c1, sort_c3*/,
      InferTensorType /*sort_c2*/]> {
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
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>):
        %predicate = stablehlo.compare GT, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %predicate : tensor<i1>
    }) {
      dimension = 0 : i64,
      is_stable = true
    } : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<2x3xi64>)
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*sort_i1*/,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension /*sort_i2*/,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable /*sort_i3*/
  );

  let results = (outs Variadic<HLO_Tensor>);

  let regions = (region SizedRegion<1>:$comparator /*sort_i4*/);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, CArg<"int64_t", "-1">:$dimension,
      CArg<"bool", "false">:$is_stable)>];

  let hasVerifier = 1;
}

def StableHLO_ReverseOp: StableHLO_ShapedInterfaceOp<"reverse",
      [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect,
       SameOperandsAndResultElementType /*reverse_c1*/,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
    DenseI64ArrayAttr:$dimensions
  );

  let hasVerifier = 1;

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $operand `,` `dims` `=` $dimensions
      attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def StableHLO_PadOp: StableHLO_ShapedInterfaceOp<"pad",
      [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect,
      SameOperandsAndResultElementType /*pad_c1*/,
      /*pad_c2, pad_i3, pad_i4, pad_i5*/
      AllMatchSameOperatorTrait<["edge_padding_low", "edge_padding_high",
      "interior_padding"], "$_self.size()", "size">,
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
    HLO_ScalarTensor:$padding_value /*pad_i2*/,
    DenseI64ArrayAttr:$edge_padding_low /*pad_i3*/,
    DenseI64ArrayAttr:$edge_padding_high /*pad_i4*/,
    DenseI64ArrayAttr:$interior_padding /*pad_i5*/
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` $padding_value `,`
      `low` `=` $edge_padding_low `,`
      `high` `=` $edge_padding_high `,`
      `interior` `=` $interior_padding
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_TransposeOp: StableHLO_ShapedInterfaceOp<"transpose",
      [ConditionallySpeculatable, NoMemoryEffect,
      HLO_CompatibleOperandsAndResultElementType, /*transpose_c1*/
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
    HLO_TensorOrPerAxisQuantizedTensor:$operand,
    DenseI64ArrayAttr:$permutation
  );
  let results = (outs HLO_TensorOrPerAxisQuantizedTensor:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` `dims` `=` $permutation
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_TriangularSolveOp: StableHLO_Op<"triangular_solve",
    [NoMemoryEffect, ConditionallySpeculatable,
     HLO_CompatibleOperandsAndResultElementType, InferTensorType]> {
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

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_ReduceWindowOp: StableHLO_Op<"reduce_window", [
      HLO_RecursivelySpeculatableIfAllInputsStatic,
      RecursiveMemoryEffects,
      SameVariadicOperandSize /*reduce_window_c1*/,
      SingleBlockImplicitTerminator<"ReturnOp">,
      InferTensorType /*reduce_window_c1, reduce_window_c14, reduce_window_c15, reduce_window_c16*/]> {
  let summary = "ReduceWindow operation";
  let description = [{
    Applies a reduction function `body` to windows of `inputs` and `init_values`
    and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window

    Example:
    ```mlir
    %result = "stablehlo.reduce_window"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      window_dimensions = array<i64: 2, 1>,
      window_strides = array<i64: 4, 1>,
      base_dilations = array<i64: 2, 1>,
      window_dilations = array<i64: 3, 1>,
      padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*reduce_window_i1*/,
    Variadic<HLO_Tensor>:$init_values /*reduce_window_i2*/,
    DenseI64ArrayAttr:$window_dimensions /*reduce_window_i3*/,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the operand dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<DenseI64ArrayAttr>:$window_strides /*reduce_window_i4*/,
    OptionalAttr<DenseI64ArrayAttr>:$base_dilations /*reduce_window_i5*/,
    OptionalAttr<DenseI64ArrayAttr>:$window_dilations /*reduce_window_i6*/,
    OptionalAttr<I64ElementsAttr>:$padding /*reduce_window_i7*/
  );

  let results = (outs Variadic<HLO_Tensor>);

  let regions = (region SizedRegion<1>:$body /*reduce_window_i8*/);

  let hasVerifier = 1;


  // Builder for non-variadic version of the operation.
  let builders = [
    OpBuilder<(ins "Type":$result_type, "Value":$operand,
      "Value":$init_value,
      "DenseI64ArrayAttr":$window_dimensions,
      "DenseI64ArrayAttr":$window_strides,
      "DenseI64ArrayAttr":$base_dilations,
      "DenseI64ArrayAttr":$window_dilations,
      "DenseIntElementsAttr":$padding),
    [{
      build($_builder, $_state, TypeRange(result_type), ValueRange(operand),
            ValueRange(init_value), window_dimensions, window_strides,
            base_dilations, window_dilations, padding);
    }]>,
    OpBuilder<(ins "ValueRange":$operands,
      "ValueRange":$init_values,
      "DenseI64ArrayAttr":$window_dimensions,
      "DenseI64ArrayAttr":$window_strides,
      "DenseI64ArrayAttr":$base_dilations,
      "DenseI64ArrayAttr":$window_dilations,
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
      dimensions = array<i64: 1>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrPerAxisQuantizedTensorOrToken>:$results
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
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*optimization_barrier_c1*/]> {
  let summary = "OptimizationBarrier operation";
  let description = [{
    Ensures that the operations that produce the `operand` are executed before any
    operations that depend on the `result` and prevents compiler transformations
    from moving operations across the barrier. Other than that, the operation is
    an identity, i.e. `result` = `operand`.

    See:
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
    HLO_StaticDimensionTensor:$shape,
    StableHLO_RngDistributionAttr:$rng_distribution
  );

  let results = (outs HLO_PredIntOrFpRankedTensor:$result);

  let assemblyFormat = [{
    $a `,` $b `,` $shape `,` `distribution` `=` $rng_distribution
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_RngBitGeneratorOp : StableHLO_Op<"rng_bit_generator",
    [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput,
     NoMemoryEffect]> {
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
      [], TensorOf<[HLO_Float, HLO_QuantizedInt, HLO_PerAxisQuantizedInt]> /*uniform_quantize_i1*/,
      TensorOf<[HLO_QuantizedInt, HLO_PerAxisQuantizedInt]>> { /*uniform_quantize_c1*/
  let summary = "UniformQuantize operation";
  let description = [{
    Performs element-wise conversion of floating-point tensor or quantized
    tensor `operand` to a quantized tensor `result` according to the
    quantization parameters defined by the `result` type.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#uniform_quantize

    Example:
    ```mlir
    %result = stablehlo.uniform_quantize %operand : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>
    ```
  }];

  let hasVerifier = 1;
}

def StableHLO_UniformDequantizeOp : StableHLO_UnaryElementwiseOp<"uniform_dequantize",
      [InferTensorType], TensorOf<[HLO_QuantizedInt, HLO_PerAxisQuantizedInt]> /*uniform_dequantize_i1*/,
      HLO_FpTensor> { /*uniform_dequantize_c1, uniform_dequantize_c2*/
  let summary = "UniformDequantize operation";
  let description = [{
    Performs element-wise conversion of quantized tensor `operand` to a
    floating-point tensor `result` according to the quantization parameters
    defined by the `operand` type.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#uniform_dequantize

    Example:
    ```mlir
    %result = stablehlo.uniform_dequantize %operand : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2xf32>
    ```
  }];
}

def StableHLO_ReducePrecisionOp : StableHLO_Op<"reduce_precision",
      [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect,
       Elementwise,
       HLO_CompatibleOperandsAndResultType /*reduce_precision_c1*/]> {
  let summary = "ReducePrecision operation";
  let description = [{
    Performs element-wise conversion of `operand` to another floating-point type
    that uses `exponent_bits` and `mantissa_bits` and back to the original
    floating-point type and produces an `output` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_precision

    Example:
    ```mlir
    %output = stablehlo.reduce_precision %operand, format = e5m10 : tensor<6xf64>
    ```
  }];
  let arguments = (ins
    HLO_FpOrQuantizedIntTensor:$operand, /*reduce_precision_i1*/
    ConfinedAttr<I32Attr, [IntPositive]>:$exponent_bits, /*reduce_precision_c2, reduce_precision_i2*/
    ConfinedAttr<I32Attr, [IntNonNegative]>:$mantissa_bits /*reduce_precision_c3, reduce_precision_i3*/
  );
  let results = (outs HLO_FpOrQuantizedIntTensor:$output);

  let assemblyFormat = [{
    $operand `,` `format` `=` custom<ExponentMantissa>($exponent_bits, $mantissa_bits)
      attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($output))
  }];
}

def StableHLO_RealDynamicSliceOp: StableHLO_ShapedInterfaceOp<
      "real_dynamic_slice",
      [ConditionallySpeculatable, NoMemoryEffect,
       AllElementTypesMatch<["operand", "result"]>,
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

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_DynamicPadOp: StableHLO_ShapedInterfaceOp<"dynamic_pad",
      [ConditionallySpeculatable, NoMemoryEffect,
      AllElementTypesMatch<["operand", "padding_value", "result"]> /*dynamic_pad_c1*/,
      AllTypesMatch<["edge_padding_low", "edge_padding_high", "interior_padding"]> /*dynamic_pad_c2*/,
      AllRanksMatch<["operand", "result"]> /*dynamic_pad_c4*/]> {
  let summary = "DynamicPad operation";
  let description = [{
    This operation is functionally identical to
    [pad](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad)
    https://github.com/openxla/stablehlo/pull/2306#discussion_r1595669709
    op, but with `edge_padding_low`, `edge_padding_high` and `interior_padding`
    specified dynamically as values.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_pad

    Example:
    ```mlir
    %edge_padding_low = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
    %edge_padding_high = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
    %interior_padding = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
    %result = stablehlo.dynamic_pad %operand, %padding_value,
                %edge_padding_low, %edge_padding_high, %interior_padding
                : (tensor<2x3xi64>, tensor<i64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x9xi64>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_pad_i1*/,
    HLO_ScalarTensor:$padding_value /*dynamic_pad_i2*/,
    HLO_StaticDimensionTensor:$edge_padding_low /*dynamic_pad_i3*/,
    HLO_StaticDimensionTensor:$edge_padding_high /*dynamic_pad_i4*/,
    HLO_StaticDimensionTensor:$interior_padding /*dynamic_pad_i5*/
  );
  let results = (outs HLO_Tensor:$result);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_DynamicGatherOp: StableHLO_Op<"dynamic_gather",
      [HLO_SpeculatableIfAllInputsStaticAndShapeConstant, NoMemoryEffect,
      InferTensorTypeWithReify]> {
  let summary = "DynamicGather operation";
  let description = [{
    This operation is functionally identical to
    [gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather)
    op, with the `slice_sizes` specified dynamically as an operand.

    Example:
    ```mlir
    %slice_sizes = stablehlo.constant dense<[1, 2, 2]> : tensor<3xi64>
    %result = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2, 3],
        collapsed_slice_dims = [0],
        start_index_map = [0, 2],
        index_vector_dim = 2>,
      indices_are_sorted = false
    } : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_gather_i1*/,
    HLO_IntTensor:$start_indices /*dynamic_gather_i2*/,
    HLO_Static1DIntTensor:$slice_sizes /*dynamic_gather_i3*/,
    StableHLO_GatherDimensionNumbers:$dimension_numbers /*dynamic_gather_i3, dynamic_gather_i4, dynamic_gather_i5, dynamic_gather_i6, dynamic_gather_i7*/,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted /*dynamic_gather_i8*/
  );
  let results = (outs HLO_Tensor);
}

def StableHLO_DynamicConvOp : StableHLO_Op<"dynamic_conv",
      [HLO_SpeculatableIfAllInputsStaticAndShapeConstant, NoMemoryEffect]> {
  let summary = "DynamicConv operation";
  let description = [{
    This operation is functionally identical to
    [convolution](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution)
    op, but the padding is specified dynamically via `padding`.

    Example:
    ```mlir
    %padding = stablehlo.constant dense<2> : tensor<2x2xi64>
    %result = "stablehlo.dynamic_conv"(%lhs, %rhs, %padding) {
      window_strides = array<i64: 4, 4>,
      lhs_dilation = array<i64: 2, 2>,
      rhs_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>, tensor<2x2xi64>) -> tensor<1x2x2x1xi64>
    ```
  }];

  let arguments = (ins
       HLO_Tensor:$lhs, /*dynamic_conv_i1*/
       HLO_Tensor:$rhs, /*dynamic_conv_i2*/
       HLO_Static2DIntTensor:$padding, /*dynamic_conv_i3*/
       OptionalAttr<DenseI64ArrayAttr>:$window_strides, /*dynamic_conv_i4*/
       OptionalAttr<DenseI64ArrayAttr>:$lhs_dilation, /*dynamic_conv_i5*/
       OptionalAttr<DenseI64ArrayAttr>:$rhs_dilation, /*dynamic_conv_i6*/
       OptionalAttr<DenseBoolArrayAttr>:$window_reversal, /*dynamic_conv_i7*/
       StableHLO_ConvDimensionNumbers:$dimension_numbers, /*dynamic_conv_i8...dynamic_conv_i16*/
       ConfinedAttr<I64Attr, [IntPositive]>:$feature_group_count, /*dynamic_conv_c21, dynamic_conv_i17*/
       ConfinedAttr<I64Attr, [IntPositive]>:$batch_group_count, /*dynamic_conv_c22, dynamic_conv_i18*/
       OptionalAttr<StableHLO_PrecisionConfigAttr>:$precision_config /*dynamic_conv_i19*/
    );
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasWindowReversal() {
      auto reversal = getWindowReversal();
      return reversal.has_value() && llvm::any_of(reversal.value(), [](bool v) { return v; });
    }
  }];
}

#endif // STABLEHLO_DIALECT_STABLEHLO_OPS
