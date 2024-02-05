# RFC: StableHLO quantization for reduction ops

Status: Approved<br/>
Initial version: 06/22/2023<br/>
updated: 07/13/2023: Minor refactoring of the examples.<br/>
Last updated: 08/11/2023: Revision of the proposal to introduce an
attribute to capture accumulation type.<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/1664)

## Version log

* 06/22/2023: Initial version.
* 07/13/2023: Fixed typo in code blocks, header indentation.
* 08/11/2023: Revision of the proposal to introduce an attribute to capture
              accumulation type.
* 08/25/2023: The additional attribute is redundant.

## Introduction

The [reduce](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce)
op, for non-quantized types, has constraints like

```python
(C2) element_type(inputs...) = element_type(init_values...) = element_type(results...).
(C6) body has type tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ..., tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>) where Ei = element_type(inputs[i]).
```

which constrained the signature of reduce op and its associated reducer function
`body` to have the same element types for `inputs`, `results` and arguments and
return for `body`. For reducer function performing an accumulative operation
like add, this means that the the result of accumulation can overflow in which
case the result will be implementation defined (e.g.,
        [saturated](https://en.wikipedia.org/wiki/Saturation_arithmetic) or
        [wrap around](https://en.wikipedia.org/wiki/Integer_overflow)).  From
the conversation with customers it seems a reasonable behavior for non quantized
data types. However, with quantized data types, such loss in precision is not
acceptable and hence the motivation is to perform the accumulation in some
higher data type.

The RFC introduces the following proposal, emerged out of discussion in the
[thread](https://github.com/openxla/stablehlo/pull/1538#issuecomment-1599476906)
, along with their tradeoffs.

The proposal optionally allows the reducer block to express the computation in a
different element type (preferably wider accumulation type) than the one used in
reduce op's ops arguments and return type. For illustrative purposes, in the
following example, the operand element type `tensor<!quant.uniform<ui8:f32,
input_scale:input_zp>>`  is different from the element type for reduction
region's block arguments.  Similarly, the element type of the reduce op's
result `!quant.uniform<ui8:f32, output_scale:output_zp>>` is different from
that of block return (`tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>`).

```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
    ^reduce_computation(
            %lhs: tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>,
            %rhs: tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>):
        %add = "stablehlo.add"(%lhs, %rhs)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>,
               tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>)
            -> tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>
        "stablehlo.return"(%add)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>) -> ()
    }) {
        dimensions = dense<1> : tensor<i64>
    } : (tensor<5 x 1 x !quant.uniform<ui8:f32, input_scale:input_zp>>,
         tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>)
    -> tensor<5 x !quant.uniform<ui8:f32, output_scale:output_zp>>
```

### Semantics

If (1) the input operand type is different from the reduction block argument
type or (2) the op result type is different from the reduction block return
type, there will be an implicit type conversion defined by either
`stablehlo.convert`, `stablehlo.uniform_quantize`, or
`stablehlo.uniform_dequantize`. When the types are not differnet, i.e., when (1)
and (2) does not hold true, then no implicit convertion is needed.

For example,

 | Implicit type conversion op       | element type of operand or block return | element type of block argument or op return |
 |-----------------------------------|-----------------------------------------|---------------------------------------------|
 | (A) `stablehlo.uniform_quantize`  | quantized tensor                        | quantized tensor                            |
 | (B) `stablehlo.uniform_quantize`  | floating point                          | quantized tensor                            |
 | (C) `stablehlo.uniorm_dequantize` | quantized tensor                        | floating point                              |
 | (D) `stablehlo.convert`           | floating-point                          | integer                                     |
 | (E) `stablehlo.convert`           | integer                                 | floating-point                              |
 | (F) `stablehlo.convert`           | floating-point                          | floating-point                              |
 | (G) `stablehlo.convert`           | integer                                 | integer                                     |
 | (H) `stablehlo.convert`           | complex                                 | complex                                     |

At this point there is no use for cases other than (A), (F), and (G).  My
proposal here would be to address (A), (F), and (G) only.  Note that the (F)
partially addresses
[Decide on mixed precision](https://github.com/openxla/stablehlo/issues/369)
for reduce op in that it allows the input or init value to differ from the
corresponding block arguments w.r.t the precision of floating-point types.
However, the mixed precision implementation in HLO seems more detailed in the
following sense:

* [Decide on mixed precision](https://github.com/openxla/stablehlo/issues/369)
allows `inputs` and `init_values` to differ in floating-point precision.
Whereas, the current proposal considers them to have the same element type.
* [Decide on mixed precision](https://github.com/openxla/stablehlo/issues/369)
allows the element type of block arguments to differ from that of the block
return value. The current proposal considers them to have the same element type.
* There are other ops (than reduce) which need support for mixed precision (here
is the [list of ops](https://github.com/tensorflow/tensorflow/blob/1d69ba72834b963b72075a82c10959f6bb74e473/tensorflow/compiler/xla/service/hlo_verifier.cc#L1681-L1714)).

Having said that, my proposal would be to treat the above ticket separately.

## Appendix

To provide an estimate of specification changes needed to implement the
proposal, I have attempted to provide the blueprint here.

### Revised specification of reduce op

Here we include only the relevant portions of the spec with the proposed update.

#### Semantics

...

More formally, `results...[j0, ..., jR-1] =
reduce_implicit_convert(reduce(input_slices_converted),
        type(func_outputs(body)...), type(results...)))` where:

* `input_slices = inputs...[j0, ..., :, ..., jR-1]`, where `:` are inserted
  at `dimensions`.
* `input_slices_converted = reduce_implicit_convert(input_slices...,
        type(inputs...), type(func_inputs(body)...)`.
* `reduce(input_slices_converted) = exec(schedule)` for some binary tree
  `schedule` where:
  * `exec(node) = body(exec(node.left), exec(node.right))`.
  * `exec(leaf) = leaf.value`.
* `schedule` is an implementation-defined full binary tree whose in-order
  traversal consists of:
  * `input_slices_converted...[index]` values, for all `index` in
    `index_space(input_slices_converted)` in the ascending lexicographic order
    of `index`.
  * Interspersed with an implementation-defined amount of
  `reduce_implicit_convert(init_values..., type(init_values...), type(func_inputs(body)[:len(func_inputs(body)//2)])...)`
    at implementation-defined positions.

#### Constraints

* (C?) `same(shape(inputs...))`.
* (C?) `element_type(inputs...) = element_type(init_values...)`.
* (C?) `baseline_element_type(inputs...) = baseline_element_type(results...)`.
* (C?) `body` has type `tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ...,`
       `tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>)` where
       `is_integer(element_type(inputs[i])) = is_integer(element_type(E[i]))` or
       `is_float(element_type(inputs[i])) = is_float(element_type(E[i]))` or
       `is_complex(element_type(inputs[i])) = is_complex(element_type(E[i]))` or
       `is_quantized(element_type(inputs[i])) = is_quantized(element_type(E[i]))`.
* (C?) `shape(results...) = shape(inputs...)` except that the dimension
  sizes of `inputs...` corresponding to `dimensions` are not included.

`reduce_implicit_convert` is defined as

```python
def reduce_implicit_convert(x: Value, source_type: Type, destination_type:
        Type):
    if source_type == destination_type:
        return x
    if is_quantized(source_type) and is_quantized(destination_type):
        return quantize(x, destination_type)
    return convert(x, destination_type)
```

The above specification of `reduce` op can be used to define the specification
of other ops as shown below. As before, we are only presenting the relevant
portions of the spec which needs modification.

### Revised specification of  reduce_window op

#### Constraints

* (C?) `element_type(inputs...) = element_type(init_values...)`.
* (C?) `baseline_element_type(inputs...) = baseline_element_type(results...)`.
* (C?) `body` has type `(tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ...,`
       `tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>)` where
       `is_integer(element_type(inputs[i])) = is_integer(element_type(E[i]))` or
       `is_float(element_type(inputs[i])) = is_float(element_type(E[i]))` or
       `is_complex(element_type(inputs[i])) = is_complex(element_type(E[i]))` or
       `is_quantized(element_type(inputs[i])) = is_quantized(element_type(E[i]))`.

### Revised specification of select_and_scatter op

This op originally takes two function arguments `select` and `scatter`. As the
`select` function is supposed to perform a non-accumulative operation, we may
not need additional conversion functions associated with `select`. But the
`scatter` function needs be accompanied with `input_conversion` and
`output_conversion` functions.

#### Constraints

<!-- markdownlint-disable line-length -->
* (C1) `element_type(operand) = element_type(source)`.
* (C3) `element_type(init_value) = element_type(operand)`.
* (C?) `baseline_element_type(inputs...) = baseline_element_type(results...)`.
* (C10) `scatter` has type `(tensor<E>, tensor<E>) -> tensor<E>` where
       `is_integer(element_type(operand)) = is_integer(element_type(E))` or
       `is_float(element_type(operand)) = is_float(element_type(E))` or
       `is_complex(element_type(operand)) = is_complex(element_type(E))` or
       `is_quantized(element_type(operand)) = is_quantized(element_type(E))`.
<!-- markdownlint-enable line-length -->

### Action Plan

I propose to follow the action plan (order matters):

* Update the specification of ReduceOp, ReduceWindowOp, and SelectAndScatterOp
  op, taking the accumulation type into account, via [open
  pr](https://github.com/openxla/stablehlo/pull/1538).
* Finalize the quantized specification of AllReduceOp, BatchNormTrainingOp,
  BatchNormGradOp and ReduceScatterOp, whose semantics depend on ReduceOp,
  via [open ticket](https://github.com/openxla/stablehlo/issues/1666).
* Spec the behavior of `precision_config` in DotGeneralOp. [open
issue](https://github.com/openxla/stablehlo/issues/755)
* Consider adding `precision_config` in reduction op.  `precision_config`,
  currently used for `dot_general` and `convolution`, to override the precision
  specified by the input parameters, allowing the choice of low precision vs
  high precision computation. We should consider adding `precision_config` to
  all reduction based op as well. [need a ticket for this]
* Consider adding `accumulation_type` to `dot_general`/`convolution op`. The
  attribute seems beneficial for ops like `dot_general` and `convolution` which
  does not have an explicit reduction function. [need a ticket for this item].

## Summary of previous proposals

For completeness of the presentation, let me provide the proposals which are
evaluated previously and help shape the current proposal.

### Re-scale input to accumulation type

This option is the simplest from the POV for specification of quantized `reduce`
op. This is adding `stablehlo.uniform_quantize`ops before and after reduce op
which operates on the "accumulator" type.

```mlir
%widen = "stablehlo.uniform_quantize"(%input)
    : (tensor<... x !quant.uniform<ui8:f32, ...>>) -> tensor<... x !quant.uniform<i32:f32, ...>>

%reduce = "stablehlo.reduce"(%widen) {
    ^reduce_computation(%lhs: !quant.uniform<i32:f32, ...>, %rhs: !qunat.uniform<i32:f32, ...>):
        // reduce_computation_block
    }
    : (tensor<... x !quant.uniform<i32:f32, ...>>) -> tensor<... x !quant.uniform<i32:f32, ...>>

%narrowed = "stablehlo.uniform_quantize"(%reduce)
    : (tensor<... x !quant.uniform<i32:f32, ...>>) -> tensor<... x !quant.uniform<ui8:f32, ...>>
```

#### Tradeoffs

* (+) An advantage of this option is that we only need minor changes to the
  specification (i.e. to allow quantized types).
* (-) The compiler must pattern match 3 operations and map them into some
  internal representation before their compilation or execution.
* (-) The compiler must ensure that the `stablehlo.uniform_quantize` (or
  `stablehlo.convert` in the case of `bf16` or `f16`) is not folded before the
  backend matches the pattern.
  [for more information](https://github.com/openxla/stablehlo/pull/1538#issuecomment-1599476906)

This proposal should be avoided because it is hard to control the transformation
which might disrupt the pattern to be matched.

### Introduce on-the-fly type conversions

Proposes addition two regions in reduce op to (1) convert the input type to the
type of the `body` function argument and (2) convert the result type of the
`body` function to the output type. Following is the code snippet with the
proposed syntax of reduce op:

```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
    ^input_conversion(
            %input: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>):
        %input_rescaled = "stablehlo.uniform_quantize"(%input)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>)
            -> tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>
        "stablehlo.return"(%input_rescaled)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>) -> ()

    }, {
    ^reduce_computation(
            %lhs: tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>,
            %rhs: tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>):
        %add = "stablehlo.add"(%lhs, %rhs)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>,
               tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>)
            -> tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>
        "stablehlo.return"(%add)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>) -> ()
    }, {
    ^output_conversion(
            %intermediate_result: tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>):
        %output_rescaled = "stablehlo.uniform_quantize"(%intermediate_result)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>)
            -> tensor<!quant.uniform<ui8:f32, output_scale:output_zp>>
        "stablehlo.return"(%output_rescaled)
            : (tensor<!quant.uniform<ui8:f32, output_scale:output_zp>>) -> ()
    }) {
        dimensions = dense<...> : tensor<1xi64>
    } : (tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>,
         tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>)
    -> tensor<... x !quant.uniform<ui8:f32, output_scale:output_zp>>
```

Here we will informally propose the semantics of the additional functions
`input_conversion` and `output_conversion` introduced.

```python
+----------+  +--------+ +--------+    +----------+  +--------+ +--------+
|init_value|  |input[0]| |input[1]|    |init_value|  |input[2]| |input[3]|
+----------+  +--------+ +--------+    +----------+  +--------+ +--------+
    |             |          |               |           |          |
+----------+  +--------+ +--------+    +----------+  +--------+ +--------+
|input     |  |input   | |input   |    |input     |  |input   | |input   |
|convert   |  |convert | |convert |    |convert   |  |convert | |convert |
+----------+  +--------+ +--------+    +----------+  +--------+ +--------+
      \      /           /                   \      /           /
      +-------+         /                    +-------+         /
      |compute|        /                     |compute|        /
      +-------+       /                      +-------+       /
             \       /                              \       /
              +-------+                              +-------+
              |compute|                              |compute|
              +-------+                              +-------+
                     \___________           ___________/
                                 \         /
                                  +-------+
                                  |compute|
                                  +-------+
                                      |
                                  +-------+
                                  |output |
                                  |convert|
                                  +-------+
```

### Tradeoffs

* (+) Enables programmers to program at (almost) baremetal. If the hardware
  can support reduction computation in wider type (e.g. in the SIMD
  instruction set, we typically do widening/compute/narrowing within the
  kernel to save the memory bandwidth), the programmer can explicitly request
  for that.
* (-) The disadvantage of this representation is that the syntax is more
  verbose and requires significant changes to the specification.
* (-) The extra input/output conversion blocks are surplus information.  The
  intent of conversion blocks is to capture the accumulation type needed to
  compute the accumulative operation on. The specification would benefit if the
  intent can be expressed succinctly.

### Introduce accumulation type attribute

Instead of using additional input and output conversion blocks, use a type
attribute `accumulation type` to capture the accumulation type. As an example,

```mlir
%0 = stablehlo.reduce(%arg0 init: %arg1) across dimensions = [0] {
    accumulation_type = tensor<!quant.uniform<i32:f32, 3.400000e+01:16>>
} : (tensor<16x!quant.uniform<i8:f32, 3.400000e+01:16>>, tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>) -> tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>
    reducer(%arg2: tensor<!quant.uniform<i32:f32, 3.400000e+01:16>>, %arg3: tensor<!quant.uniform<i32:f32, 3.400000e+01:16>>)  {
     %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i32:f32, 3.400000e+01:16>>
     stablehlo.return %1 : tensor<!quant.uniform<i32:f32, 3.400000e+01:16>>
    }

// using tablegen specification like
// OptionalAttr<TypeAttrOf<HLO_Tensor>>:$accumulation_type
```

Note that the main difference between this option and the previous option  is
that the input and output conversion blocks are no longer used and their intent
is specified via the `accumulation_type` attribute. However, the reducer block
needs to express the computation in accumulation type only.

This options is discarded because, for reduce op, the additional attribute seems
redundant and can be inferred based on the differences in element type of
operand and reduction block arguments (as described in the current proposal).
