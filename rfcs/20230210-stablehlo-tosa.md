# RFC: Align StableHLO and TOSA arithmetic

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | |
| **Author(s)** | Eric Kunze (eric.kunze@arm.com), Dominic Symes (dominic.symes@arm.com), Tom Cooksey (tom.cooksey@arm.com) |
| **Updated**   |                                               |

## Objective

Align the operations of StableHLO and TOSA.

## Proposal

* Define the arithmetic operation of StableHLO to enable implementation via
TOSA operators.
* Enable implementation of fully quantized networks on systems without
floating-point hardware such as
[Arm Ethos-U55](https://developer.arm.com/documentation/102420/0200/Programmers-model/Operators-and-performance/Operations).

## Background

Both StableHLO as well as TOSA provide a set of operations for processing ML
models.
They operate at adjacent points in the compilation stack, providing stability
points for implementations.
StableHLO is aimed closer to the framework and compiler, while TOSA is focused
closer to hardware implementations and smaller systems.

Fully quantized networks have converted all of their operators into integer
based operations.
Integer operations are generally lower power, and often faster than doing the
equivalent floating-point operation.
The silicon area required to implement integer operations is also significantly
smaller, leading some accelerators to omit floating-point circuitry compltely,
particularly in the mobile/embedded market.
The weights and activations when quantized down to 8 or 16 bits also require
less memory traffic due to their smaller size.

Fully integer networks also fully guarantee reproducibility across completely
independent implementations.
They can be tested bit accurately, and there is no need for fuzz testing to
capture possible errors.

### TOSA

Tensor Operator Set Architecture ([TOSA](https://mlplatform.org/tosa)) provides
a set of whole-tensor operations commonly employed by Deep Neural Networks. The
intent of TOSA is to enable a variety of implementations running on a diverse
range of processors, with the results at the TOSA level consistent across those
implementations. This consistency includes systems such as microcontrollers,
where the cost of a floating-point multiplier is a significant burden in area
and power.

TOSA requires that results are bit accurately defined for integer operations.
This requirement is in place to ensure portability across implementations.

TOSA provides a conformance test suite and reference model to verify that
implementations match the expected behavior.

There is a TOSA dialect representing the specification in the MLIR repository,
evolving to match the TOSA specification.

TOSA Operations don't accept quantized tensors, quantization has been moved to
explicit RESCALE operations. RESCALE is an integer multiply, add, and shift,
operations widely available. Explicit zero points where needed for asymmetric
quantization.

### Example TOSA expanded quantized operation

This is an example of an expanded quantized add operation.
Before performing the integer addition, both inputs must be brought to a common scale.
Multiple options are available for the multiplier and shift.
This allows the scaling to be customized to match the behavior of the original framework.

### Framework operator

```c
%framework_sum = framework.add %a %b :
    (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform(i8:f32, .075:-1)>>)
    -> tensor<2x2x!quant.uniform(i8:f32, 0.15:-1)>
```

### TOSA quantized implementation

```c
%scaled_a = tosa.rescale %a
    {input_zp = -1 : i32, multiplier = [1431655765 : i32], shift = [13 : i32]}} :
    (tensor<2x2xi8>) -> tensor<2x2xi32>
%scaled_b = tosa.rescale %b
    {input_zp = -1 : i32, multiplier = [1073741824 : i32], shift = [11 : i32]}} :
    (tensor<2x2xi8>) -> tensor<2x2xi32>
%sum = tosa.add %scaled_a %scaled_b :
    (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
%scaled_sum = tosa.rescale %sum
    {{input_zp = -1 : i32, multiplier = [[1073741824 : i32], shift = [50 : i32]}} :} :
    (tensor<2x2xi32>) -> tensor<2x2xi8>>
```

In this instance, the quantization has been removed from the input and output
tensors. The scale and zero point that are in the quantization parameters has
been moved as the multiplier and shift included in the RESCALE operation's
attributes. The TOSA RESCALE operation does not use the quantization parameters,
only the multiplier and shift attributes that are part of the operation. Each
step along this path is well defined in terms of integer arithmetic, therefore
all implementations will return the same result. If a floating-point operation
was used in rescale, it opens up the possibility of bit errors and leads to
issues for implementation on integer only devices. This can lead to varying
results between different devices/implementations. With TOSA's guarantee,
portability across implementations is much easier.

### Quantization in StableHLO

StableHLO accepts quantized tensors for operations, but the behavior in the
existing specification is not defined. This RFC references the appendix of the
FP8 in XLA [RFC](https://github.com/openxla/xla/discussions/22) to get an
understanding of how quantization is implemented in StableHLO.

As described in the FP8 RFC, the quantization itself does not appear to
significantly differ from what TOSA does, much of the work required to align is
making the underlying behavior explicit. Once the behavior is explicit, then
comparisons to TOSA quantization behavior can be made and it should be possible
to align the behavior. One key holdup in the current StableHLO design is that
the quantization scale is kept as a floating-point number. This adds ambiguity
for quantized values, as the value makes it impossible to be described as a
fully integer based operation that could be run on an integer only processor,
and prevents bit accurate checking of the results.

## Design Proposal

To improve the experience for developers using StableHLO to express fully
quantized models, align the arithmetic operations with TOSA and it's handling of
quantized operations to ensure networks compiled using StableHLO return the same
results across implementations.

### Make quantized StableHLO operations explicit

Define which operations allow quantized input/output tensors. Do all tensors
which accept integers accept quantized tensors?

Convert the quantization parameters to an integer multiplier and shift.

Add a new StableHLO operator rescale, which performs the same function as the
TOSA RESCALE operator.

Add a pass to replace the quantized tensors with integer tensors. A TOSA pass
exists for this purpose, so could be adapted to perform the same operation with
StableHLO operators.

For each operator that accepts a quantized tensor, define the behavior using
methods which align with TOSA operations. Define any intermediate data sizes
that would have effects on the results (i.e. accumulator size for convolution)

## Open Issues

**TOSA does not currently define FP8 operations.**

As described in the StableHLO FP8 RFC, they may be able to be treated as uniform
quantized tensors within the dialects. This can serve as another alignment point
but is not addressed in this RFC.

**Dynamic quantization scaling during training.**

Also as described in the FP8 RFC, dynamic scaling is a desirable feature for
training. Neither StableHLO nor TOSA currently support dynamic scale support.
Both dialects would need updating, although it does not appear that the
operators in the specifications would need to significantly change.

**Floating-point compound operator accuracy requirements.**

The next draft of the TOSA specification includes a proposed error bound for
compound operations such as convolution. This could be another point for
collaboration.

## References

1. [TOSA Specification](https://www.mlplatform.org/tosa/tosa_spec.html)
2. [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
3. [StableHLO FP8 RFC](https://github.com/openxla/xla/discussions/22)
4. [Arm Ethos-U55 NPU](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u55)
