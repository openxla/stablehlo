# Quantization in StableHLO

StableHLO quantization follows the [LiteRT quantization
specification](https://ai.google.dev/edge/litert/models/quantization_spec),
using a uniform quantization scheme with support for both per-tensor and
per-axis quantization. It inherits its type expression from MLIR's [Quant
dialect](https://mlir.llvm.org/docs/Dialects/QuantDialect/), providing a
standardized way to represent quantized data types.

## Uniform Quantization

Quantization is a technique to optimize machine learning models by
converting floating-point numbers (like those used in original models)
into lower-precision integers. This reduces memory usage and speeds up
computations, making models more efficient for deployment on devices with
limited resources.

One common approach is uniform quantization, where we map the
floating-point values to integers that are evenly spaced.  To do this, we
use two key quantization parameters:

- **Scale:** This determines the step size between consecutive quantized values.
A smaller scale means the quantized values are closer together, allowing for
finer-grained representation.
- **Zero Point:** This integer value represents the real value 0 in the
quantized space.

The relationship between the original floating-point value (`real_value`) and
the quantized integer value (`quantized_value`) in uniform quantization is:

```python
real_value = scale * (quantized_value + zero_point)
```

### Per-tensor Quantization

In per-tensor quantization, a single scale and zero point are used for all the
values within the tensor. A per-tensor quantized type is expressed in StableHLO
as:

```mlir
!quant.uniform<storage_type:expressed_type, scale:zero_point>
```

**Example**: `!quant.uniform<i8:f32, 0.01:50>`

This represents an 8-bit integer (`i8`) used to store a 32-bit floating-point
number (`f32`) using a scale of `0.01` and a zero point of `50`.

### Per-axis Quantization

Per-axis quantization offers a more fine-grained approach compared to per-tensor
quantization. Instead of using a single scale and zero point for the entire
tensor, per-axis quantization assigns separate scales and zero points to slices
along a specific dimension `quantized_dimension` of the tensor. This is
particularly useful when values vary significantly across different dimensions,
allowing for better preservation of information and accuracy.

Consider a tensor t with dimensions sizes `[4, 3, 2, 1]`. We choose to quantize
this tensor along the second dimension (`quantized_dimension = 1`). This means
we'll have three slices (since the second dimension has a size of 3), each with
its own scale and zero point:

```python
t[:, 0, :, :]: This slice gets scale[0] and zero_point[0].
t[:, 1, :, :]: This slice gets scale[1] and zero_point[1].
t[:, 2, :, :]: This slice gets scale[2] and zero_point[2].
```

In StableHLO, per-axis quantized type is expressed as:

```mlir
!quant.uniform<storage_type:expressed_type:quantized_dimension, {scale:zero_point, scale:zero_point, ...}>
```

where the length of the `scale:zero_point` matches the number of slices along
the `quantized_dimension` of the containing tensor.

**Example**:  `!quant.uniform<i8:f32:1, {0.2:20, 0.1:10, 0.3:30}>`

**Note**: StableHLO will soon support _sub-channel quantization_, which allows
for quantization along a subset of dimensions. This feature is currently in
development and will be available in a future release. For more information,
see the [design doc](https://discourse.llvm.org/t/rfc-supporting-sub-channel-quantization-in-mlir/82694).

## Quantization Passes in StableHLO

StableHLO provides several compiler passes which allow for different
transformations and optimizations related to quantization, giving you
flexibility in how you handle quantized models. These passes are:

### `stablehlo-legalize-qdq-to-quantized-op` [code](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/StablehloLegalizeQDQToQuantizedOp.cpp)

This pass fuses a common pattern in quantized models, a dequantize operation
followed by a floating-point operation, and finally a quantize operation, into
a single quantized operation.

**Example:**

```mlir
// Before the pass
func.func @add(%arg0: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = stablehlo.uniform_dequantize %arg0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
  %1 = stablehlo.abs %0 : tensor<16x16xf32>
  %2 = stablehlo.uniform_quantize %1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %2 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// After the pass
func.func @add(%arg0: tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>> {
  %0 = stablehlo.uniform_dequantize %arg0 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
  %1 = stablehlo.abs %0 : tensor<16x16xf32>
  %2 = stablehlo.abs %arg0 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
  %3 = stablehlo.uniform_quantize %1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
  return %2 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
}
```

### stablehlo-legalize-quantized-op-to-qdq [code](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/StablehloLegalizeQuantizedOpToQDQ.cpp)

This pass does the opposite of the previous pass. It decomposes a quantized
operation into its equivalent sequence of dequantize, floating-point operation,
          and quantize operations.

**Example:**

```mlir
// Before the pass
func.func @add(%arg0: tensor<!quant.uniform<i8:f32,1.0:0>>, %arg1: tensor<!quant.uniform<i8:f32,2.0:1>>) ->  tensor<!quant.uniform<i8:f32,3.0:2>> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32,1.0:0>>, tensor<!quant.uniform<i8:f32,2.0:1>>) -> tensor<!quant.uniform<i8:f32,3.0:2>>
  func.return %0 : tensor<!quant.uniform<i8:f32,3.0:2>>
}

// After the pass
func.func @add(%arg0: tensor<!quant.uniform<i8:f32, 1.000000e+00>>, %arg1: tensor<!quant.uniform<i8:f32, 2.000000e+00:1>>) -> tensor<!quant.uniform<i8:f32, 3.000000e+00:2>> {
  %0 = stablehlo.uniform_dequantize %arg0 : (tensor<!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<f32>
  %1 = stablehlo.uniform_dequantize %arg1 : (tensor<!quant.uniform<i8:f32, 2.000000e+00:1>>) -> tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  %3 = stablehlo.uniform_quantize %2 : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 3.000000e+00:2>>
  return %3 : tensor<!quant.uniform<i8:f32, 3.000000e+00:2>>
}
```

### stablehlo-legalize-quant-to-math [code](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/StablehloLegalizeQuantToMath.cpp)

This pass converts StableHLO operations on quantized types into equivalent
operations on integer types. It essentially implements the quantization
arithmetic using standard mathematical operations. This decompsition is useful
for systems that do not support quantization natively, but can still use the
quantization arithmetic to express the semantics of quantized models.

**Example:**

```mlir
// Before the pass
func.func @add(%arg0: tensor<!quant.uniform<i8:f32,1.0:0>>, %arg1: tensor<!quant.uniform<i8:f32,2.0:1>>) ->  tensor<!quant.uniform<i8:f32,3.0:2>> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32,1.0:0>>, tensor<!quant.uniform<i8:f32,2.0:1>>) -> tensor<!quant.uniform<i8:f32,3.0:2>>
  func.return %0 : tensor<!quant.uniform<i8:f32,3.0:2>>
}

// After the pass
func.func @add(%arg0: tensor<i8>, %arg1: tensor<i8>) -> tensor<i8> {
  %0 = stablehlo.convert %arg0 : (tensor<i8>) -> tensor<f32>
  %cst = stablehlo.constant dense<0.333333343> : tensor<f32>
  %1 = chlo.broadcast_multiply %0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %2 = chlo.broadcast_add %1, %cst_0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = stablehlo.round_nearest_even %2 : tensor<f32>
  %4 = stablehlo.convert %3 : (tensor<f32>) -> tensor<i32>
  %5 = stablehlo.convert %arg1 : (tensor<i8>) -> tensor<f32>
  %cst_1 = stablehlo.constant dense<0.666666686> : tensor<f32>
  %6 = chlo.broadcast_multiply %5, %cst_1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %cst_2 = stablehlo.constant dense<1.33333337> : tensor<f32>
  %7 = chlo.broadcast_add %6, %cst_2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %8 = stablehlo.round_nearest_even %7 : tensor<f32>
  %9 = stablehlo.convert %8 : (tensor<f32>) -> tensor<i32>
  %c = stablehlo.constant dense<2> : tensor<i32>
  %10 = chlo.broadcast_add %4, %9 : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %11 = chlo.broadcast_subtract %10, %c : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %c_3 = stablehlo.constant dense<-128> : tensor<i32>
  %c_4 = stablehlo.constant dense<127> : tensor<i32>
  %12 = stablehlo.clamp %c_3, %11, %c_4 : tensor<i32>
  %13 = stablehlo.convert %12 : (tensor<i32>) -> tensor<i8>
  return %13 : tensor<i8>
}
```

## stablehlo-quant-legalize-to-tosa-rescale [code](https://github.com/openxla/stablehlo/blob/main/stablehlo/conversions/tosa/transforms/StablehloQuantLegalizeToTosaRescale.cpp)

StableHLO offers the capability to legalize quantized operations to their
corresponding representations in the [TOSA
dialect](https://mlir.llvm.org/docs/Dialects/TOSA/). This legalization
facilitates compatibility and interoperability between StableHLO and TOSA.  This
pass strategically converts StableHLO quantized operations into a combination of
StableHLO and TOSA operations, with the TOSA dialect primarily employed for the
`rescale` operation. The `tosa.rescale` op plays a crucial role in adjusting the
scale and zero point of quantized values, enabling accurate representation of
quantized data within the TOSA framework.

**Example:**

Consider the following StableHLO code snippet:

```mlir
// Before the pass
func.func @add(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// After the pass
func.func @add(%arg0: tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  %r0 = "tosa.rescale"(%0) {double_round = true, input_zp = -1 : i32, multiplier = array<i32: 1431655765>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 13>} : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<2x2xi32>
  %r1 = "tosa.rescale"(%1) {double_round = true, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2xi32>

  %add = "stablehlo.add"(%r0, %r1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>

  %r2 = "tosa.rescale"(%add) {double_round = true, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = -1 : i32, per_channel = false, scale32 = true, shift = array<i8: 50>} : (tensor<2x2xi32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>

  return %r2 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}
```

The `stablehlo.add` operation now works on integer types (`i32`), and the
`tosa.rescale` operations are strategically placed to manage the necessary
scaling and zero-point adjustments for accurate quantization.

## tosa-rescale-legalize-to-stablehlo [code](https://github.com/openxla/stablehlo/blob/main/stablehlo/conversions/tosa/transforms/TosaRescaleLegalizeToStablehlo.cpp)

This pass rewrites TOSA rescale operations to StableHLO primitive math
operations. One of the main use cases for this pass is to allow the StableHLO
interpreter to evaluate programs containing TOSA rescale operations.

## Evaluating Quantized Programs

The [StableHLO reference
interpreter](https://github.com/openxla/stablehlo/blob/main/docs/reference.md)
can efficiently execute programs containing quantized operations. To achieve
this, it first lowers the program to an equivalent representation using only
integer operations. This lowering process involves a series of compiler passes
that transform the program before interpretation.

Essentially, the interpreter leverages the `stablehlo-legalize-quant-to-math`
pass to convert quantized operations into their corresponding integer arithmetic
implementations. This pass introduces CHLO broadcast operations for handling
scale multiplication/division and zero-point addition.  To ensure compatibility
with the StableHLO interpreter, these CHLO operations are then legalized to
StableHLO operations. This introduces shape-related operations that are
subsequently canonicalized and optimized using a series of canonicalization
passes.

The complete sequence of passes involved in this lowering process is as follows:

```mlir
stablehlo-legalize-quant-to-math
chlo-legalize-to-stablehlo
canonicalize
shape-legalize-to-stablehlo
stablehlo-canonicalize-dynamism
```

**Note:** There is an ongoing effort to improve the efficiency of this lowering
process. You can track the progress in this [open
issue](https://github.com/openxla/stablehlo/issues/2390).
