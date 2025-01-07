# StableHLO Quantization

## Quantization Types in StableHLO

Quantization is a technique to optimize machine learning models by
converting floating-point numbers (like those used in original models)
into lower-precision integers. This reduces memory usage and speeds up
computations, making models more efficient for deployment on devices with
limited resources.

StableHLO quantization follows the [LiteRT quantization
specification](https://ai.google.dev/edge/litert/models/quantization_spec),
using a uniform quantization scheme with support for both per-tensor and
per-axis quantization. It inherits its type expression from MLIR's [Quant
dialect](https://mlir.llvm.org/docs/Dialects/QuantDialect/), providing a
standardized way to represent quantized data types.

Uniform quantization maps floating-point values to integers using a uniform step
size, resulting in evenly spaced quantized values. This is achieved through an
affine ralationship using two key quantization parameters.

Uniform quantization simplifies the representation of floating-point numbers by
mapping them to integers that are evenly spaced. This mapping is achieved
through an affine transformation that uses two key parameters: **scale** and
**zero point**. The scale determines determines the step size between
consecutive quantized values. A smaller scale means the quantized values are
closer together. The zero point defines the integer value that represents zero
in the original floating-point space.

The relationship between the original floating-point value (`real_value`) and
the quantized integer value (`quantized_value`) in uniform quantization is:

```python
real_value = scale * (quantized_value - zero_point)
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

Consider a tensor t with dimensions sizes `[4, 3, 2]`. We choose to quantize
this tensor along the second dimension (`quantized_dimension = 1`). This means
we'll have three slices (since the second dimension has a size of 3), each with
its own scale and zero point:

```python
t[:, 0, :]: This slice gets scale[0] and zero_point[0].
t[:, 1, :]: This slice gets scale[1] and zero_point[1].
t[:, 2, :]: This slice gets scale[2] and zero_point[2].
```

In StableHLO, per-axis quantized type is expressed as:

```mlir
!quant.uniform<storage_type:expressed_type:quantized_dimension, {scale0:zero_point0, scale1:zero_point1, ...}>
```

where the length of the `scale:zero_point` matches the number of slices along
the `quantized_dimension` of the containing tensor.

**Example**:  `tensor<4x3x2x!quant.uniform<i8:f32:1, {0.2:20, 0.1:10, 0.3:30}>>`

**Note**: StableHLO will soon support _sub-channel quantization_, which allows
for quantization along a subset of dimensions. This feature is currently in
development and will be available in a future release. For more information,
see the [design doc](https://discourse.llvm.org/t/rfc-supporting-sub-channel-quantization-in-mlir/82694).

## Quantization Passes in StableHLO

StableHLO provides several compiler passes which allow for different
transformations and optimizations related to quantization, giving you
flexibility in how you handle quantized models. These passes are:

### `stablehlo-legalize-qdq-to-quantized-op`

This pass fuses a common pattern in quantized models, a dequantize operation
followed by a floating-point operation, and finally a quantize operation, into
a single quantized operation. [details](https://openxla.org/stablehlo/generated/stablehlo_passes#-stablehlo-legalize-qdq-to-quantized-op)

### stablehlo-legalize-quantized-op-to-qdq

This pass does the opposite of the previous pass. It decomposes a quantized
operation into its equivalent sequence of dequantize, floating-point operation,
and quantize operations.
[details](https://openxla.org/stablehlo/generated/stablehlo_passes#-stablehlo-legalize-quantized-op-to-qdq)

### stablehlo-legalize-quant-to-math

This pass converts StableHLO operations on quantized types into equivalent
operations on integer types. It essentially implements the quantization
arithmetic using standard mathematical operations. This decompsition is useful
for systems that do not support quantization natively, but can still use the
quantization arithmetic to express the semantics of quantized models.
[details](https://openxla.org/stablehlo/generated/stablehlo_passes#-stablehlo-legalize-quant-to-math)

## stablehlo-quant-legalize-to-tosa-rescale

StableHLO offers the capability to legalize quantized operations to their
corresponding representations in the [TOSA
dialect](https://mlir.llvm.org/docs/Dialects/TOSA/). This legalization
facilitates compatibility and interoperability between StableHLO and TOSA.  This
pass strategically converts StableHLO quantized operations into a combination of
StableHLO and TOSA operations, with the TOSA dialect primarily employed for the
`rescale` operation. The `tosa.rescale` op plays a crucial role in adjusting the
scale and zero point of quantized values, enabling accurate representation of
quantized data within the TOSA framework.
[details](https://openxla.org/stablehlo/generated/stablehlo_tosa_passes#-stablehlo-quant-legalize-to-tosa-rescale)

## tosa-rescale-legalize-to-stablehlo

This pass rewrites TOSA rescale operations to StableHLO primitive math
operations. One of the main use cases for this pass is to allow the StableHLO
interpreter to evaluate programs containing TOSA rescale operations.
[details](https://openxla.org/stablehlo/generated/stablehlo_tosa_passes#-tosa-rescale-legalize-to-stablehlo)

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

## Quantized Test Cases

StableHLO provides a comprehensive suite of quantized test cases to validate the
correctness and behavior of quantized operations. These test cases serve as unit
tests, covering various StableHLO operations in quantized scenarios.

A typical example of a quantized test case looks like

```mlir
func.func @main() -> tensor<11xf32> {
    %operand_0 = stablehlo.constant dense<...> : tensor<11xf32>
    %operand_1 = stablehlo.constant dense<...> : tensor<11xf32>
    %golden = stablehlo.constant dense<...> : tensor<11xf32>

    %0 = stablehlo.uniform_quantize %operand_0 : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.3>>
    %1 = stablehlo.uniform_quantize %operand_1 : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.3>>

    %2 = stablehlo.add %1, %0 : tensor<11x!quant.uniform<i8:f32, 0.3>>

    %result = stablehlo.uniform_dequantize %2 : (tensor<11x!quant.uniform<i8:f32, 0.3>>) -> tensor<11xf32>

    %4 = stablehlo.custom_call @check.eq(%golden, %result) : (tensor<11xf32>, tensor<11xf32>) -> tensor<i1>

    return %3 : tensor<11xf32>
  }
```

and includes:

- **Input data:** Representative input values for the operation.
- **Golden output:** The expected output of the operation when applied to the
input data, complying with the [StableHLO reference
interpreter](https://github.com/openxla/stablehlo/blob/main/docs/reference.md)
and the [HLO
evaluator](https://github.com/openxla/xla/tree/main/xla/hlo/evaluator).

These test cases are valuable for:

- **Validating StableHLO quantization:** Ensuring that the quantization behavior
of StableHLO operations aligns with the expected results.
- **Cross-validation:** Comparing the behavior of StableHLO quantization with
other implementations or frameworks.
- **Debugging and development:**  Aiding in the development and debugging of new
quantization features or optimizations.
