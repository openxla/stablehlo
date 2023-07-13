# RFC: StableHLO quantization for reduction ops

Status: Review<br/>
Initial version: 06/22/2023<br/>
Last updated: 07/13/2023<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/1664)

## Version log

* 06/22/2023: Initial version.
* 07/13/2023: Fixed typo in code blocks, header indentation.

## Introduction

The [reduce](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce)
op, for non-quantized types, has constraints like

```python
(C2) element_type(inputs...) = element_type(init_values...) = element_type(results...).
(C6) body has type tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ..., tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>) where Ei = element_type(inputs[i]).
```

which constrained the signature of reduce op and its associated reducer function
`body` to have the same element types for `inputs`, `results` and arguments and
return for `body`. For reducer function performing an accumulative operation like
add, this means that the the result of accumulation can overflow in which case
the result will be implementation defined (e.g.,
[saturated](https://en.wikipedia.org/wiki/Saturation_arithmetic) or
[wrap around](https://en.wikipedia.org/wiki/Integer_overflow)).
From the conversation with customers it seems a reasonable behavior for non
quantized data types. However, with quantized data types, such loss in precision
is not acceptable and hence the motivation is to perform the accumulation in
some higher data type.

The RFC highlights some of the options emerged out of discussion in the
[thread](https://github.com/openxla/stablehlo/pull/1538#issuecomment-1599476906)
along with their tradeoffs. The proposal option #1 looks promising at this
point, but we are open to further discussion on this.

## Option 1: Introduce additional conversion functions

[The thread](https://github.com/openxla/stablehlo/pull/1538#issuecomment-1599476906)
discuses an option, proposed by @loganchien, on how to achieve the structural
changes as mentioned above. We note that some of the examples/diagrams presented
here are borrowed from an internal doc @loganchien authored.

The proposed options introduces on-the-fly type conversions, which (1) convert
the input type to the type of the `body` function argument and (2) convert the
result type of the `body` function to the output type. Following is the code
snippet with the proposed syntax of reduce op:

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

### Semantics

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

### Semantics of `input_conversion` block

The `input_conversion` block is applied selectively to the leaf nodes of a
schedule tree as shown in above diagram. Note that the `input_conversion` cannot
be applied to the non-leaf nodes of the schedule tree.

### Semantics of `output_conversion` block

The `output_conversion` block is applied just after the `result` for a particular
index is computed as shown in the above diagram.

Please refer to the [formal spec](#revised-specification-of-reduce-op) of the proposed
reduce op.

### Implementation details

From the implementation POV of the proposed spec, we note that
`input_conversion` or `output_conversion` can very well be optional with
default values as identity functions. For example, the following code snippet

```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
    ^reduce_computation(
            %lhs: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>,
            %rhs: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>):
        %add = "stablehlo.add"(%lhs, %rhs)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>,
               tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>)
            -> tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>
        "stablehlo.return"(%add)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>) -> ()
    }) {
        dimensions = dense<...> : tensor<1xi64>
    } : (tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>,
         tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>)
    -> tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>
```

should be interpreted as

```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
    ^input_conversion(
            %input: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>):
        "stablehlo.return"(%input)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>) -> ()

    }, {
    ^reduce_computation(
            %lhs: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>,
            %rhs: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>):
        %add = "stablehlo.add"(%lhs, %rhs)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>,
               tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>)
            -> tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>
        "stablehlo.return"(%add)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>) -> ()
    }, {
    ^output_conversion(
            %intermediate_result: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>):
        "stablehlo.return"(%intermediate_result)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>) -> ()
    }) {
        dimensions = dense<...> : tensor<1xi64>
    } : (tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>,
         tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>)
    -> tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>
```

Note that with default values, the  input/result type of `reduce` op matches
with the argument or the result type of the `reduce_computation`, including the
quantization parameters.

Also, note that the relative order of `input_conversion` or `output_conversion`
w.r.t the `reduce_computation` can be used to identify the appropriate
conversion function when any one of `input_conversion` or `output_conversion` is
missing.

The existing pretty printing is currently producing the following output
`stablehlo.reduce(%input init: %init_value) applies stablehlo.add across
dimensions = [1] : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>`. IMO,
modifying the above format, with the default conversion function, will create
clutter. My proposal here is to follow the existing pretty printing when the
conversion functions are "not provided". In the event, the conversion functions
are explicitly provided, then the pretty printers will fall back to default
generic printing,
**even if the explicitly provided conversion functions are identity function**:
To avoid identification of identity functions which could be tricky in general.

### Tradeoffs

* (+) Enables programmers to program at (almost) baremetal. If the hardware
  can support reduction computation in wider type (e.g. in the SIMD
  instruction set, we typically do widening/compute/narrowing within the
  kernel to save the memory bandwidth), the programmer can explicitly request
  for that.
* (-) The disadvantage of this representation is that the syntax is more
  verbose and requires significant changes to the specification.

## Option 2: re-scale input to accumulation type

This option is the simplest from the POV for specification of quantized `reduce`
op. This is adding `stablehlo.uniform_quantize` and `stablehlo.dequantize` ops
respectively before and after reduce op which operates on the "accumulator"
type.

```mlir
%widen = "stablehlo.uniform_quantize"(%input)
    : (tensor<... x !quant.uniform<ui8:f32, ...>>) -> tensor<... x !quant.uniform<i32:f32, ...>>

%reduce = "stablehlo.reduce"(%widen) {
    ^reduce_computation(%lhs: !quant.uniform<i32:f32, ...>, %rhs: !qunat.uniform<i32:f32, ...>):
        // reduce_computation_block
    }
    : (tensor<... x !quant.uniform<i32:f32, ...>>) -> tensor<... x !quant.uniform<i32:f32, ...>>

%narrowed = "stablehlo.uniform_dequantize"(%reduce)
    : (tensor<... x !quant.uniform<i32:f32, ...>>) -> tensor<... x !quant.uniform<ui8:f32, ...>>
```

### Tradeoffs

* (+) An advantage of this option is that we only need minor changes to the
  specification (i.e. to allow quantized types).
* (-) The compiler must pattern match 3 operations and map them into some
  internal representation before their compilation or execution.
* (-) The compiler must ensure that the `stablehlo.uniform_quantize` (or
  `stablehlo.convert` in the case of `bf16` or `f16`) is not folded before the
  backend matches the pattern.
  [for more information](https://github.com/openxla/stablehlo/pull/1538#issuecomment-1599476906)

## Option 3: allow accumulator type to be different from input type

This is another option we considered which does not fly well because of limited
expressibility. Adding it just for completeness purposes.
The idea here is to convey the accumulator type using the `init_value` operand
of `reduce` op. The code snippet for `reduce` looks like:

```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
    ^reduce_computation(
            %elem: tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>,
            %acc: tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>):
        %elem_rescaled = "stablehlo.uniform_quantize"(%elem)
            : (tensor<!quant.uniform<ui8:f32, input_scale:input_zp>>)
            -> tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>
        %add = "stablehlo.add"(%elem_rescaled, %acc)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>,
               tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>)
            -> tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>
        "stablehlo.return"(%0)
            : (tensor<!quant.uniform<i32:f32, accum_scale:accum_zp>>) -> ()
    }) {
        dimensions = dense<1> : tensor<1xi64>
    } : (tensor<... x !quant.uniform<ui8:f32, input_scale:input_zp>>,
         tensor<... x !quant.uniform<i32:f32, accum_scale:accum_zp>>)
    -> tensor<... x !quant.uniform<i32:f32, accum_scale:accum_zp>>
```

In this option, the `init_value` type and the `result` type can be different
from the input type. The first argument of the compute block is fixed for the
traversed element and the second argument is fixed for the intermediate
(accumulation) result.

### Tradeoffs

* (+) Make the accumulation type explicit in the IR.
* (-) This representation imposes a limitation on the evaluation order.
  Since we can’t express the computation between two intermediate (accumulation)
  results, we can not arbitrarily insert `init_value` and start the
  computation at an arbitrary location. The following shows the restricted
  evaluation order with the method.

```python
+----------+       +--------+ +--------+ +--------+ +--------+
|init_value|       |input[0]| |input[1]| |input[2]| |input[3]|
+----------+       +--------+ +--------+ +--------+ +--------+
           \        /         /          /          /
           +-------+         /          /          /
           |compute|        /          /          /
           +-------+       /          /          /
                \         /          /          /
                 +-------+          /          /
                 |compute|         /          /
                 +-------+        /          /
                       \         /          /
                        +-------+          /
                        |compute|         /
                        +-------+        /
                              \         /
                               +-------+
                               |compute|
                               +-------+
```

## Open Question

### Should we restrict the proposal #1 to quantized types only?

The above proposal #1 of introducing the additional functions is theoretically
not limited to quantized `reduce` op, but also can be applied to `reduce` op with
non-quantized types. For example,

```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
  ^input_conversion(%arg0: tensor<bf16>):
    %0 = "stablehlo.convert"(%arg0): (tensor<bf16>) -> (tensor<f32>)
    "stablehlo.return"(%0) : (tensor<f32>) -> (tensor<f32>)
  }, {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) ->
    tensor<f32>
    "stablehlo.return"(%0) : (tensor<f32>) -> ()
  }, {
  ^output_conversion(%arg0: tensor<f32>):
    %0 = "stablehlo.convert"(%arg0): (tensor<f32>) -> (tensor<bf16>)
    "stablehlo.return"(%0) : (tensor<bf16>) -> (tensor<bf16>)
  }) {
  dimensions = dense<1> : tensor<1xbf16>
} : (tensor<1x6xbf16>, tensor<bf16>) -> tensor<1xbf16>
```

However, it is not clear how such operations will be lowered to other IR
representations, like HLO, which does not support such additional computation
blocks. IMO there is no additional benefit to support such conversion
functions for regular type given that there already exists infrastructure
(backend support, lowering passes) to support regular types w/o conversion
functions. My proposal here would be to restrict the support to only quantized
types.

## Appendix

To provide an estimate of specification changes needed to implement option #1
I have attempted to provide the blueprint here.

### Revised specification of reduce op

#### Semantics

Applies a reduction functions `input_conversion`, `body`, and
`output_conversion` to `inputs` and `init_values` along the `dimensions` and
produces `results` tensors.

The order of reductions is implementation-defined, which means that `body` and
`init_values` must form a monoid to guarantee that the operation produces the
same results for all inputs on all implementations. However, this condition
doesn't hold for many popular reductions. E.g. floating-point addition for
`body` and zero for `init_values` don't actually form a monoid because
floating-point addition is not associative.

More formally, `results...[j0, ..., jR-1] =
map(output_conversion, reduce(input_slices_converted))` where:

* `input_slices = inputs...[j0, ..., :, ..., jR-1]`, where `:` are inserted
  at `dimensions`.
* `input_slices_converted = map(input_conversion, input_slices...)`.
* `reduce(input_slices_converted) = exec(schedule)` for some binary tree
  `schedule` where:
  * `exec(node) = body(exec(node.left), exec(node.right))`.
  * `exec(leaf) = leaf.value`.
* `schedule` is an implementation-defined full binary tree whose in-order
  traversal consists of:
  * `input_slices_converted...[index]` values, for all `index` in
    `index_space(input_slices_converted)` in the ascending lexicographic order
    of `index`.
  * Interspersed with an implementation-defined amount of `init_values`
    at implementation-defined positions.

#### Inputs

| Label | Name                | Type                                         | Constraints |
|-------|---------------------|----------------------------------------------|-------------|
| (I?)  | `inputs`            | variadic number of tensors                   |             |
| (I?)  | `init_values`       | variadic number of 0-dimensional tensors     |             |
| (I?)  | `dimensions`        | 1-dimensional tensor constant of type `si64` |             |
| (I?)  | `input_conversion`  | function                                     |             |
| (I?)  | `body`              | function                                     |             |
| (I?)  | `output_conversion` | function                                     |             |

#### Outputs

| Name      | Type                       | Constraints |
|-----------|----------------------------|-------------|
| `results` | variadic number of tensors |             |

#### Constraints

* (C?) `same(shape(inputs...))`.
* (C?) `element_type(inputs...) = element_type(init_values...)`.
* (C?) `baseline_element_type(inputs...) = baseline_element_type(results...)`.
* (C?) `0 < size(inputs) = size(init_values) = size(results) = N`.
* (C?) `0 <= dimensions < rank(inputs[0])`.
* (C?) `is_unique(dimensions)`.
* (C?) `input_conversion` has type `tensor<E0>, ..., tensor<EN-1> ->
       (tensor<E'0>, ..., tensor<E'N-1>)` where `Ei = element_type(inputs[i])`.
* (C?) `body` has type `tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ...,`
       `tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>)` where
       `Ei = element_type(output_types(input_conversion)[i])`.
* (C?) `output_conversion` has type `tensor<E0>, ..., tensor<EN-1> ->
       (tensor<E'0>, ..., tensor<E'N-1>)` where
       `E'i = element_type(results[i])`.
* (C?) `element_type(output_types(input_conversion)...) =
       element_type(input_types(output_conversion)...)`.
* (C?) `shape(results...) = shape(inputs...)` except that the dimension
  sizes of `inputs...` corresponding to `dimensions` are not included.

The above specification of `reduce` op can be used to define the specification
of other ops as shown below. For brevity, we are only presenting the relevant
portions of the spec which needs modification.

### Revised specification of  reduce_window op

#### Semantics

Applies a reduction functions `input_conversion`, `body`, and
`output_conversion` to windows of `inputs` and `init_values` and produces
`results`.

...

More formally,
`results...[result_index] = reduce(windows, init_values, axes(inputs...),
        input_conversion, body, output_conversion)`
where:
....

#### Inputs

| Label | Name                | Type     |
|-------|---------------------|----------|
| (I?)  | `input_conversion`  | function |
| (I8)  | `body`              | function |
| (I?)  | `output_conversion` | function |

#### Constraints

* (C?) `element_type(inputs...) = element_type(init_values...)`.
* (C?) `baseline_element_type(inputs...) = baseline_element_type(results...)`.
* (C?) `input_conversion` has type `tensor<E0>, ..., tensor<EN-1> ->
       (tensor<E'0>, ..., tensor<E'N-1>)` where `Ei = element_type(inputs[i])`.
* (C?) `body` has type `tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ...,`
       `tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>)` where
       `Ei = element_type(output_types(input_conversion)[i])`.
* (C?) `output_conversion` has type `tensor<E0>, ..., tensor<EN-1> ->
       (tensor<E'0>, ..., tensor<E'N-1>)` where
       `E'i = element_type(results[i])`.
* (C?) `element_type(output_types(input_conversion)...) =
       element_type(input_types(output_conversion)...)`.

### Revised specification of select_and_scatter op

This op originally takes two function arguments `select` and `scatter`. As the
`select` function is supposed to perform a non-accumulative operation, we may
not need additional conversion functions associated with `select`. But the
`scatter` function needs be accompanied with `input_conversion` and
`output_conversion` functions.

#### Semantics

Scatters the values from the `source` tensor using `scatter` based on the
outcome of `reduce_window` of the `input` tensor using `select` and produces
a `result` tensor.

More formally:
...

* `result[result_index] = reduce([source_values], [init_value], [0],
        input_conversion, scatter, output_conversion)`
 where:
 ...

#### Inputs

| Label | Name                | Type     |
|-------|---------------------|----------|
| (I8)  | `input_conversion`  | function |
| (I8)  | `scatter`           | function |
| (I8)  | `output_conversion` | function |

#### Constraints

<!-- markdownlint-disable line-length -->
* (C1) `element_type(operand) = element_type(source)`.
* (C3) `element_type(init_value) = element_type(operand)`.
* (C?) `baseline_element_type(inputs...) = baseline_element_type(results...)`.
* (C?) `input_conversion` has type `tensor<E> -> (tensor<E'>)` where
       `Ei = element_type(operand)`.
* (C10) `scatter` has type `(tensor<E>, tensor<E>) -> tensor<E>` where
        `E = element_type(output_types(input_conversion))`.
* (C?) `output_conversion` has type `tensor<E> -> (tensor<E'>)` where
       `E'i = element_type(result)`.
* (C?) `element_type(output_types(input_conversion)) =
       element_type(input_types(output_conversion))`.
* (C11) `shape(operand) = shape(result)`.
<!-- markdownlint-enable line-length -->
