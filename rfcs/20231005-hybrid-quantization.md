# RFC: Hybrid quantized ops for weight-only quantization

Status: Approved<br/>
Initial version: 10/05/2023<br/>
Last updated: 10/31/2023<br/>

## Version log

* 10/05/2023: Initial version.
* 10/17/2023: Minor fixes and add proposed operation semantics.

## Introduction

This RFC proposes to extend the convolution and dot_general ops to allow
differing element types for their operands and define their semantics to
represent weight-only quantization. This work is related to the more general
topic of [mixed precision](https://github.com/openxla/stablehlo/issues/369), but
does not aim to address it in its entirety.

Weight-only quantization is a quantization scheme which is often used to reduce
the size of models that are memory-bound. A weight-only quantized graph accepts
input in float and weight in quantized type, dequantizes the weight and performs
float computation. Even though weight-only quantization can already be expressed
using a combination of uniform_dequantize op and float-only op, we propose to
extend the semantics of some float-only ops as a representation of weight-only
quantization because of the following reasons.

* It aligns well with the existing behavior of the corresponding HLO ops which
allow operand types to differ and upcast operands to the highest-precision type
in such a case.
* Being able to express weight-only natively in StableHLO, it does not rely on
pattern matching by backend to avoid patterns that might result in bad
performance.
* Constant and dequantize can unintentionally be folded in downstreams and
hybrid op removes concern on constant folding. We can also consider inserting
optimization_barrier in between to prevent constant folding, but this requires
upstream frameworks to embed optimization_barrier and downstreams to pattern
match additional optimization_barrier. It is hard to guarantee these
requirements on use-cases across various frameworks and hardwares.

## Examples

Here are examples of hybrid quantized convolution and dot_general. Hybrid
quantized ops will get float input(or lhs) and quantized weight(or rhs) as
operands and output float results.

```mlir
%conv = "stablehlo.convolution"(%input_f, %weight_q) ... : (tensor<...xf32>, !quant.uniform<i8:f32, scale:zp>) -> tensor<...xf32>>
```

```mlir
%dot = "stablehlo.dot_general"(%lhs_f, %rhs_q) ... : (tensor<...xf32>, !quant.uniform<i8:f32, scale:zp>) -> tensor<...xf32>>
```

## Proposed spec changes

We propose to modify operation semantics and a few constraints on quantized
convolution and dot_general to represent weight-only quantization using hybrid
op.

### hybrid_dequantize_then_op semantics

We propose to define `hybrid_dequantize_then_op` semantics as part of quantization
computations.

* `hybrid_dequantize_then_op` is used to specify weight-only quantization for
hybrid op which accepts lhs in floating-point and rhs in quantized types. It
dequantizes quantized inputs into their expressed types and performs computation
in float. Element type of float lhs tensor and expressed type of quantized rhs
tensor should be identical.

```python
def hybrid_dequantize_then_op(op, lhs, rhs):
  assert(is_float(lhs) and is_quantized(rhs) and element_type(lhs) == expressed_type(rhs))
  return op(lhs, dequantize(rhs))
```

### convolution

#### Operation semantics for hybrid op

For hybrid quantized types, performs `hybrid_dequantize_then_op( lambda lhs,
rhs: convolution(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
window_reversal, input_batch_dimension, input_feature_dimension,
input_spatial_dimensions, kernel_input_feature_dimension,
kernel_output_feature_dimension, kernel_spatial_dimensions,
output_batch_dimension, output_feature_dimension, output_spatial_dimensions,
feature_group_count, batch_group_count, precision_config), lhs, rhs)`.

#### Current constraints

* If the operation uses quantized tensors:
  * (C28) `is_quantized_tensor(lhs) and is_quantized_tensor(rhs) and
    is_quantized_tensor(result)`.
  * (C29) `storage_type(lhs) =  storage_type(rhs)`.
  * (C30) `expressed_type(lhs) = expressed_type(rhs) = expressed_type(result)`.
  * (C31) If `is_per_tensor_quantized(rhs)`,
    then `is_per_tensor_quantized(result)`.
  * (C32) If `is_per_axis_quantized(rhs)`, then
    `quantization_dimension(rhs) = kernel_output_feature_dimension`.
  * (C33) If `is_per_axis_quantized(result)`, then
    `quantization_dimension(result) = output_feature_dimension`.

#### Proposed constraints

* If the operation uses quantized tensors:
  * (C28) `is_quantized(lhs) = is_quantized(result) and is_quantized(rhs)`.
  * (C29) If `is_per_axis_quantized(rhs)`,
    then `quantization_dimension(rhs) = kernel_output_feature_dimension`.
  * (C30) If `is_per_axis_quantized(result)`, then
    `quantization_dimension(result) = output_feature_dimension`.
  * If `is_quantized(lhs)`:
    * (C31) `storage_type(lhs) = storage_type(rhs)`.
    * (C32) `expressed_type(lhs) = expressed_type(rhs) = expressed_type(result)`.
    * (C33) If `is_per_tensor_quantized(rhs)`, then
      `is_per_tensor_quantized(result)`.
  * If `!is_quantized(lhs)`:
    * (C34) `element_type(lhs) = expressed_type(rhs) = element_type(result)`.

### dot_general

#### Operation semantics for hybrid op

For hybrid quantized types, performs `hybrid_dequantize_then_op( lambda lhs,
rhs: dot_general(lhs, rhs, lhs_batching_dimensions, rhs_batching_dimensions,
lhs_contracting_dimensions, rhs_contracting_dimensions, precision_config), lhs,
rhs)`.

#### Current constraints

* If the operation uses quantized tensors:
  * (C14) `is_quantized(lhs) and is_quantized(rhs) and is_quantized(result)`.
  * (C15) `storage_type(lhs) = storage_type(rhs)`.
  * (C16) `expressed_type(lhs) = expressed_type(rhs) = expressed_type(result)`.
  * (C17) `zero_points(rhs) = 0`.

#### Proposed constraints

* If the operation uses quantized tensors:
  * (C14) `is_quantized(lhs) = is_quantized(result) and is_quantized(rhs)`.
  * (C15) `zero_points(rhs) = 0`.
  * If `is_quantized(lhs)`:
    * (C16) `storage_type(lhs) = storage_type(rhs)`.
    * (C17) `expressed_type(lhs) = expressed_type(rhs) = expressed_type(result)`.
  * If `!is_quantized(lhs)`:
    * (C18) `element_type(lhs) = expressed_type(rhs) = element_type(result)`.

## Other related topics not covered in this RFC

### Dynamic Range Quantization

Dynamic range quantization(DRQ) is a different quantization scheme, which is
represented in comparable MLIR dialects using the same type signature as this proposed
weight-only quantization representation. A dynamic range quantized graph also accepts
input in float and weight in quantized type. Instead of dequantizing weights, inputs
are quantized on-the-fly based on input range and computation is done in quantized
type. To represent DRQ in StableHLO, we can consider utilizing custom call, but this
issue will be considered separately from this RFC as more discussion is needed.
