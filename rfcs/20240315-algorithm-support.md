# [RFC] Add algorithm to dot_general's attributes in the StableHLO specification

Status: Approved<br/>
Initial version: 03/15/2024<br/>
Last updated: 06/18/2024<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/2096)

## Motivation

We would like to allow explicitly selecting the algorithm used for individual
`dot_general` instructions, enforcing a well-defined numeric precision. We think
that the current precision values (`DEFAULT`, `HIGH`, `HIGHEST`) are not
specific enough for this, because they mean different things between hardware
platforms and even between GPU types.

We also want to make the API flexible enough to support additional algorithms
that might be faster on certain hardware
([bf16_6x](https://arxiv.org/pdf/1904.06376.pdf), tf32_3x, etc) that can't fit
into the limited precision_config values of `DEFAULT`, `HIGH` and `HIGHEST`. For
example "6xBF16" provides similar precision to F32, but about 2x performance on
some GPUs.

When an algorithm is not supported by an accelerator, the program should fail
instead of implicitly falling back to another. It should be the responsibility
of higher level frameworks such as JAX, to define the required algorithm for
each accelerator that is used.

## Proposed Specification change

### Programs / Types

Add `tf32` to `FloatType`, thus indirectly adding it to `TensorElementType`.

This is needed to describe the `lhs_type` and `rhs_type` of dot_general ops, as
described in the next section.

### Ops / dot_general

#### Semantics

*Add these lines:*

The `DotAlgorithm` attribute defines the main properties of the algorithm used
to implement the dot operation, which also defines the precision. If the
algorithm attribute is set, then the `precision_config` is ignored, otherwise
the `precision_config` will take effect.

`DotAlgorithm.lhs_type` and `DotAlgorithm.rhs_type` are the precisions that the
LHS and RHS of the operation are rounded to, and `DotAlgorithm.accum_type` is
`AccumType` - the accumulation type. These types are independent from the
storage types of the inputs and the output. `DotAlgorithm.lhs_component_count`,
`DotAlgorithm.rhs_component_count` and `DotAlgorithm.num_primitive_ops` apply
when we are doing an algorithm which decomposes the LHS and/or RHS into multiple
components and does multiple "primitive" dot operations on those values -
usually to emulate a higher precision (e.g.
[bf16_6x](https://arxiv.org/pdf/1904.06376.pdf), tf32_3x, etc). If this is not
the case, these values should be set to 1. If
`DotAlgorithm.allow_imprecise_accum` is true, then the implementation is allowed
to accumulate in lower precision for some steps (as per
CUBLASLT_MATMUL_DESC_FAST_ACCUM).

It is up to the implementations to decide which combinations are supported.

Example `DotAlgorithm` attributes:

```txt
// Inputs are casted to tf32, and then accumulated in f32:
{lhs_type = tf32,
 rhs_type = tf32,
 accum_type = f32,
 lhs_component_count = 1,
 rhs_component_count = 1,
 num_primitive_ops = 1,
 allow_imprecise_accum = false}


// bf16_6x: each input is decomposed to 3 bf16 components, then 6 dot operations are done on those components, and the result is accumulated in f32.
{lhs_type = bf16,
 rhs_type = bf16,
 accum_type = f32,
 lhs_component_count = 3,
 rhs_component_count = 3,
 num_primitive_ops = 6,
 allow_imprecise_accum = false}


// Inputs are (casted to) f8e5m2, and we accumulate in f32, but for some steps we may accumulate in lower precision.
{lhs_type = f8e5m2,
 rhs_type = f8e5m2,
 accum_type = f32,
 lhs_component_count = 1,
 rhs_component_count = 1,
 num_primitive_ops = 1,
 allow_imprecise_accum = true}

```

In general, it is not guaranteed that the each algorithm is supported on each
accelerator type by the consumer of the StableHLO. If a given algorithm is not
supported, an error should be raised as opposed to falling back to an
alternative.

#### Inputs

*Add these rows to the table:*

Label | Name                                 | Type                    | Constraints
----- | ------------------------------------ | ----------------------- | -----------
(I8)  | `DotAlgorithm.lhs_type`              | TensorElementType       |
(I9)  | `DotAlgorithm.rhs_type`              | TensorElementType       |
(I10) | `DotAlgorithm.accum_type`            | TensorElementType       |
(I11) | `DotAlgorithm.lhs_component_count`   | constant of type `si32` | (C20)
(I12) | `DotAlgorithm.rhs_component_count`   | constant of type `si32` | (C21)
(I6)  | `DotAlgorithm.num_primitive_ops`     | constant of type `si32` | (C22)
(I7)  | `DotAlgorithm.allow_imprecise_accum` | constant of type `bool` |

#### Constraints

*Add these constraints:*

* (C20) `DotAlgorithm.lhs_component_count >= 1`.
* (C21) `DotAlgorithm.rhs_component_count >= 1`.
* (C22) `DotAlgorithm.num_primitive_ops >= 1`.

#### Examples

*Change the example:*

```mlir
// %lhs: [
//        [[1, 2],
//         [3, 4]],
//        [[5, 6],
//         [7, 8]]
//       ]
// %rhs: [
//        [[1, 0],
//         [0, 1]],
//        [[1, 0],
//         [0, 1]]
//       ]
%result = "stablehlo.dot_general"(%lhs, %rhs) {
  dot_dimension_numbers = #stablehlo.dot<
    lhs_batching_dimensions = [0],
    rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2],
    rhs_contracting_dimensions = [1]
  >,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
  algorithm = #stablehlo.dot_algorithm<
    lhs_type = tf32,
    rhs_type = tf32,
    accum_type = f32,
    lhs_component_count = 1,
    rhs_component_count = 1,
    num_primitive_ops = 1,
    allow_imprecise_accum = false
  >
} : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
// %result: [
//           [[1, 2],
//            [3, 4]],
//           [[5, 6],
//            [7, 8]]
//          ]
```

## Alternatives considered

### Adding new Precision values instead of the algorithm attribute

The precision config is per operand, but the algorithms describe multiple
properties of the computation, some of which are not connected to the operands
(such as the accumulation type, number of dot operations to use). So we think
that it's more conceptually correct to add a separate algorithm attribute.

### Making an algorithm enum instead of a composite attribute

The algorithm attribute describes multiple properties of the computation, which
could be encoded in an enum value. But this would mean a combinatorial explosion
of the enum types, especially if we'll have examples where the "precision" of
the lhs and rhs are not the same.

### Computation type

At first, we considered adding a "computation type" instead of an algorithm. But
that would be quite limited, as it wouldn't describe the AccumType and other
possible properties of the algorithm.
