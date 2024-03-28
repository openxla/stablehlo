# [RFC] Add algorithm to dot_general's parameters in the StableHLO specification

Status: Review<br/>
Initial version: 03/15/2024<br/>
Last updated: 03/15/2024<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/2096)

## Motivation

We would like to allow explicitly selecting the algorithm used for individual
`dot_general` instructions, enforcing a well-defined numeric precision. We think
that the current precision values (`DEFAULT`, `HIGH`, `HIGHEST`) are not
specific enough for this, because they mean different things between hardware
platforms and even between GPU types.

We also want to make the API flexible enough to support additional algorithms
that might be faster on certain hardware
([6xBF16](https://arxiv.org/pdf/1904.06376.pdf), 3xTF32, etc) that can't fit
into the limited precision_config values of `DEFAULT`, `HIGH` and `HIGHEST`. For
example "6xBF16" provides similar precision to F32, but about 2x performance on
some GPUs.

When an algorithm is not supported by an accelerator, the program should fail
instead of implicitly falling back to another. It should be the responsibility
of higher level frameworks such as JAX, to define the required algorithm for
each accelerator that is used.

## Proposed Specification change

### dot_general

#### Semantics

*Add these lines:*

`algorithm` defines the main properties of the algorithm used to implement the
dot operation, which also defines the precision. If the algorithm value is set
(`!= UNSET`), then the `precision_config` is ignored.

In general the first two types in the algorithm name are `AType` and `BType` -
the precisions that the LHS and RHS of the operation are rounded to, and the
third one is `AccumType` - the accumulation type. These types are independent
from the storage types of the inputs and the output (but not all combinations
are supported). Some algorithms have other suffixes too.

This is the current list of algorithms:

* `UNSET`: There is no restriction on the algorithm, the precision is decided
  based on the `precision_config`. Use this for every type combination that
  doesn't have a specific algorithm.
* `ANY_F8_ANY_F8_F32`: The inputs are used with the full precision of their
  storage type (which must be an 8-bit floating point type - the same for both
  operands). The accumulator type is F32.
* `ANY_F8_ANY_F8_F32_FAST_ACCUM`: The inputs are used with the full precision of
    their storage type (which must be an 8-bit floating point type - the same
    for both operands). The accumulator type is formally F32, but the
    intermediate results will not periodically be promoted to a higher
    precision.
* `F16_F16_F16`: The inputs are used with / casted to F16 precision and the
  accumulator type is also F16.
* `F16_F16_F32`: The inputs are used with / casted to F16 precision and the
  accumulator type is F32.
* `BF16_BF16_BF16`: The inputs are used with / casted to BF16 precision and the
  accumulator type is also BF16.
* `BF16_BF16_F32`: The inputs are used with / casted to BF16 precision and the
  accumulator type is F32.
* `BF16_BF16_F32_X3`: An algorithm which uses three BF16_BF16_F32 dot operations
  per "tile" to achive a higher precision.
* `BF16_BF16_F32_X6`: An algorithm which uses six BF16_BF16_F32 dot operations
  per "tile" to achive a higher precision ("similar" to F32_F32_F32).
* `TF32_TF32_F32`: The inputs are casted to TF32 precision and the accumulator
  type is F32.
* `TF32_TF32_F32_X3`: An algorithm which uses 3 TF32_TF32_F32 matmuls to achieve
  a higher precision ("similar" to F32_F32_F32).
* `F32_F32_F32`: The inputs are used with / casted to F32 precision and the
  accumulator type is also F32.
* `F64_F64_F64`: The inputs are used with / casted to F64 precision and the
  accumulator type is also F64.

In general, it is not guaranteed that the each algorithm is supported on each
accelerator type by the consumer of the StableHLO. If a given algorithm is not
supported, an error should be raised as opposed to falling back to an
alternative.

*Note: In the code, the enum defs would be prefixed with
STABLEHLO_DOT_ALGORITHM_.*

#### Inputs

*Add this row to the table:*

| Label | Name | Type | Constraints |
| ----- | ----- | ----- | ----- |
| (I8)  | `algorithm` | an enum of `UNSET`, `ANY_F8_ANY_F8_F32`, `ANY_F8_ANY_F8_F32_FAST_ACCUM`, `F16_F16_F16`, `F16_F16_F32`, `BF16_BF16_BF16`, `BF16_BF16_F32`, `BF16_BF16_F32_X3`, `BF16_BF16_F32_X6`, `TF32_TF32_F32`, `TF32_TF32_F32_X3`, `F32_F32_F32`, and `F64_F64_F64` | |

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
  algorithm = #stablehlo<dot_algorithm UNSET>
} : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
// %result: [
//           [[1, 2],
//            [3, 4]],
//           [[5, 6],
//            [7, 8]]
//          ]
```

## Alternatives considered

### Adding new Precision values instead of the algorithm property

The precision config is per operand, but the algorithms describe multiple
properties of the computation, some of which are not connected to the operands
(such as the accumulation type, number of dot operations to use). So we think
that it's more conceptually correct to add a separate algorithm property.

### Making the algorithm property a struct instead of an enum

The algorithm property describes multiple properties of the computation: AType,
BType, AccumType, NumOps, and occasionally other options such as FastAccum. It
maybe easier to process if we had a "struct" to describe the algorithm, instead
of a name. For example:
`{AType=BF16, BType=BF16, AccumType=F32, NumOps=6, FastAccum=False}`
instead of `BF16_BF16_F32_X6`. This could also be
useful if we introduced algorithms which have asymmetric input precisions (but
the same input storage types), to disambiguate which operand has which
precision.

Having considered the mentioned advantages, we wanted to start by adding
something more simple for now, and just add support for a few algorithms which
are important to disambiguate. (All earlier types and precisions would remain
supported for now). Also, we think that algorithm names are easier to refer to
(although this could be probably mitigated by introducing a special syntax to
write the structs). We also think that algorithms may have more different
properties which would need new struct members, but if we use an enum, we can
just append them to the name.

### Computation type

At first, we considered adding a "computation type" instead of an algorithm. But
that would be quite limited, as it wouldn't describe the AccumType and other
possible properties of the algorithm.
