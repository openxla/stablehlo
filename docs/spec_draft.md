# StableHLO Specification Draft

## Types

Following are the supported element types in StableHLO:

  * **Integer types**
    * Signed integer with two’s complement representation. Referred to in the
    document as `si<N>`, where the bit-width N ∊ {4, 8, 16, 32, 64}.
    * Unsigned integer referred to in the document as `ui<N>`, where the
    bit-width N ∊ {4, 8, 16, 32, 64}.
  * **Boolean types** referred to in the document as `i1`. Exact
  representation of boolean types (e.g. 1 byte per boolean vs 1 bit per boolean)
  is implementation-defined.
  * **Floating-point types**
    * Single precision `f32`, double precision `f64` and half precision `f16`
    floating-points complying with [IEEE 754-2019
    format](https://ieeexplore.ieee.org/document/8766229).
    * Bfloat16 `bf16` floating-point complying with [BFloat16 format](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus).
    Provides the same number of exponent bits as `f32`, so that it matches its
    dynamic range, but with greatly reduced precision. This also ensures
    identical behavior for underflows, overflows, and NaNs. However, `bf16`
    handles denormals differently from `f32`: it flushes them to zero.
  * **Complex types** represent a pair of floating-point types. Supported ones
    are `complex<f32>` (represents a par of `f32`) and `complex<f64>`
    (represents a pair of `f64`). Exact representation of complex types
    (e.g. whether the real part or the imaginary part comes first in memory)
    is implementation-dependent.

**Tensor types** are the cornerstone of the StableHLO type system. They model
immutable n-dimensional arrays and are referred to in the document as
`tensor<SxE>` where:

  * **Shape** `S` represented as `(d0)x(d1)x...x(dR-1)` is a 1-dimensional array
  of **dimension sizes** `di`, in the increasing order of the corresponding
  **dimensions** (which are also called **axes**) 0, 1, ..., R-1.
  The size `R` of this array is called **rank**. Dimension sizes have type
  `si64` and are non-negative (dimension sizes equal to zero are allowed,
  and their meaning is described below). Ranks equal to zero are also allowed,
  and their meaning is also described below.
  * **Element type** `E` is any one of the supported element types mentioned
  above.

For example, `tensor<2x3xf32>` is a tensor type with shape `2x3` and element
type `f32`. It has two dimensions (or, in other words, two axes) whose sizes
are 2 and 3. Its rank is 2.

At the logical level, a `tensor<SxE>` maps a 1-dimensional array of **indices**
`{i0, i1, ..., iR-1}` on **elements** of type `E`. If a tensor `t` maps an index
`i` on an element `e`, we say that `t[i0, i1, ..., iR-1] = e`.

Individual indices have type `si64` and are within the range `[0, di)` defined
by the corresponding dimension. The size of the index array is equal to `R`.
At the moment, StableHLO only supports dense tensors, so each tensor has
`1*(d0)*(d1)*...*(dR-1)` elements whose indices are drawn from an
**index space** which is a Cartesian product of its dimensions. For example:
  * `tensor<2x3xf32>` has 6 elements whose indices are
    `{0, 0}`, `{0, 1}`, `{0, 2}`, `{1, 0}`, `{1, 1}` and `{1, 2}`.
  * Tensors of rank zero, e.g `tensor<f32>`, have 1 element. Such tensors are
    allowed and are useful to model scalars.
  * Tensors with dimensions of size zero, e.g. `tensor<2x0xf32>`, have
    0 elements. Such tensors are allowed and are useful in rare cases, e.g.
    to model empty slices.

**Canonical representation** of a tensor is a 1-dimensional array of elements
which correspond to indices ordered lexicographically. For example, for a
`tensor<2x3xf32>` with the following mapping from indices to elements:
`{0, 0} => 1`, `{0, 1} => 2`, `{0, 2} => 3`, `{1, 0} => 4`, `{1, 1} => 5`,
`{1, 2} => 6` - the canonical representation would be: `[1, 2, 3, 4, 5, 6]`.

Exact representation of tensors is implementation-defined. This specification
does not define in which order tensor elements are laid out in memory (e.g.
whether/when they follow the canonical order) and how individual tensor elements
in a particular order are packed together into a tensor (e.g. how these elements
are aligned, whether they are stored contiguously, etc).

## Programs

StableHLO programs consist of functions. Each function has operands and results
of supported types and a list of ops in static single-assignment (SSA) form
which is terminated by a return op which produces the results of the function.
StableHLO ops take operands and produce results.

```mlir
ml_program.func @example_func(%arg: tensor<2x3xf32>) -> tensor<2x3xf32> {
 %0 = "stablehlo.floor"(%arg) : (tensor<2x3xf32>) -> tensor<2x3xf32>
 %1 = "stablehlo.ceil"(%arg) : (tensor<2x3xf32>) -> tensor<2x3xf32>
 %2 = "stablehlo.add"(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
 ml_program.return %2 : tensor<2x3xf32>
}
```

A program is executed by passing argument values to a given function and
computing result values. Result values of a function are computed by evaluating
the graph of ops rooted in the corresponding return op. The evaluation order is
implementation-defined, as long as ops are evaluated before their uses. Possible
execution orders of the above example program are `%0` → `%1` → `%2` → `return`
or `%1` → `%0` → `%2` → `return`.

## Errors

StableHLO programs are validated through an extensive set of constraints for
individual ops, which rules out many classes of errors prior to run time.
However, error conditions are still possible, e.g. through integer overflows,
out-of-bounds accesses, etc. Unless explicitly called out, all these errors
result in implementation-defined behavior.

As an exception to this rule, floating-point exceptions in StableHLO programs
have well-defined behavior. Operations which result in exceptions defined by the
IEEE-754 standard (invalid operation, division-by-zero, overflow, underflow, or
inexact exceptions) produce default results (as defined in the standard) and
continue execution without raising the corresponding status flag; similar to
`raiseNoFlag` exception handling from the standard. Exceptions for nonstandard
operations (e.g. complex arithmetic and certain transcendental functions) are
implementation-defined.

## Constants

The section describes the constants supported in StableHLO along with their
syntax.

  * **Integer constants** use decimal notation, e.g. `123`, or hexadecimal
  notation, e.g. `ff`. Negative numbers can be used with signed integer types,
  but not with unsigned integer types.
  * **Boolean constants** `true` and `false` are both valid constants of the
  `pred` type.
  * **Floating-point constants** use decimal notation, e.g. `123.421`,
  exponential notation, e.g. `1.23421e+2`, or a more precise hexadecimal
  notation, e.g. `0x42f6d78d`.
  * **Complex constants** Complex constants are represented as a pair of
  floating-point constants of `f32` or `f64` types, e.g. `(12.34, 56.78)`,
  where the first constant is the real part, and the second constant is the
  imaginary part.
  * **Tensor constants** use NumPy notation. For example,
  `[[1, 2, 3], [4, 5, 6]]` is a constant of type `tensor<2x3xf32>` with the
  following mapping from indices to elements: `{0, 0} => 1`, `{0, 1} => 2`,
  `{0, 2} => 3`, `{1, 0} => 4`, `{1, 1} => 5`, `{1, 2} => 6`.

## Structure of an Op’s Specification

The specification of an op comprises of the following components (in the order
described below)

  * **Semantics** Semantics of the operation.
  * **Operands** Meaning of operand(s) and their type(s).
  * **Results** Meaning of the result(s) and the type(s).
  * **Constraints** Constraints on the operand(s) and the result(s).
  * **Examples** Examples demonstrating the working of the op using
    [MLIR generic syntax](https://mlir.llvm.org/docs/LangRef/#operations).

## Index of Ops
   * [abs](#stablehloabs)
   * [add](#stablehloadd)
   * [and](#stablehloand)
   * [ceil](#stablehloceil)
   * [constant](#stablehloconstant)
   * [cosine](#stablehlocosine)
   * [divide](#stablehlodivide)
   * [exponential](#stablehloexponential)
   * [floor](#stablehlofloor)
   * [log](#stablehlolog)
   * [logistic](#stablehlologistic)
   * [maximum](#stablehlomaximum)
   * [minimum](#stablehlominimum)
   * [negate](#stablehlonegate)
   * [not](#stablehlonot)
   * [or](#stablehloor)
   * [remainder](#stablehloremainder)
   * [reshape](#stablehloreshape)
   * [rsqrt](#stablehlorsqrt)
   * [sine](#stablehlosine)
   * [sqrt](#stablehlosqrt)
   * [subtract](#stablehlosubtract)
   * [tanh](#stablehlotanh)
   * [transpose](#stablehlotranspose)
   * [xor](#stablehloxor)

## stablehlo.abs

### Semantics

Performs element-wise absolute value of `operand` tensor and produces a `result`
tensor. For floating-point element types, it implements the `abs` operation from
the IEEE-754 specification.

For n-bit signed integer, the absolute value of $-2^{n-1}$ is implementation-
defined and one of the following:

  * Saturation to $2^{n-1}-1$
  * $-2^{n-1}$

### Operands

| Name      | Type                                                        |
|-----------|-------------------------------------------------------------|
| `operand` | tensor of integer, floating-point, or complex element types |

### Results

| Name     | Type                                                        |
|----------|-------------------------------------------------------------|
| `result` | tensor of integer, floating-point, or complex element types |

### Constraints

  * (C1)  `operand` and `result` have the same shape.
  * (C2)  `operand` and `result` have the same element type, except when the
  element type of the `operand` is complex type, in which case the element type
  of the `result` is the element type of the complex type (e.g. the element type
  of the `result` is `f64` for operand type `complex<f64>`).

### Examples

```mlir
// integers
// %operand: [-2, 0, 2]
%result = "stablehlo.abs"(%operand) : (tensor<3xi32>) -> tensor<3xi32>
// %result: [2, 0, 2]

// floats
// %operand: [-2.2, 0.0, 2.2]
%result = "stablehlo.abs"(%operand) : (tensor<3xf32>) -> tensor<3xf32>
// %result = [2.2, 0.0, 2.2]

// complex
// %operand: [(0.0, 1.0), (4.0, -3.0)]
%result = "stablehlo.abs"(%operand) : (tensor<2xcomplex<f64>>) -> tensor<2xf64>
// %result = [1, 5.0]
```

[Back to Ops](#index-of-ops)

## stablehlo.add

### Semantics

Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
`result` tensor. For integer element types, if the element-wise sum has an
unsigned/signed overflow/underflow, the result is implementation-defined and one
of the following:

  * mathematical result modulo $2^n$, where n is the bit width of the result,
  for unsigned overflow/underflow. For signed integer overflow/underflow, wraps
  the result around the representable range $[-2^{n-1},\ 2^{n-1} - 1]$.
  * saturation to $2^{n-1} - 1$ (or $-2^{n-1}$) for signed overflow (or signed
  underflow) and saturation to $2^n - 1$ (or $0$) for unsigned overflow (or
  unsigned underflow).

For floating-point element types, it implements the `addition` operation from
the IEEE-754 specification.

### Operands

| Name  | Type                                                        |
|-------|-------------------------------------------------------------|
| `lhs` | tensor of integer, floating-point, or complex element types |
| `rhs` | tensor of integer, floating-point, or complex element types |

### Results

| Name     | Type                                                        |
|----------|-------------------------------------------------------------|
| `result` | tensor of integer, floating-point, or complex element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// %lhs: [[1, 2], [3, 4]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[6, 8], [10, 12]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_add.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.and

### Semantics

Performs element-wise bitwise AND of two tensors `lhs` and `rhs` of integer
types and produces a `result` tensor. For boolean tensors, it computes the
logical operation.

### Operands

| Name  | Type                                       |
|-------|--------------------------------------------|
| `lhs` | tensor of integer or boolean element types |
| `rhs` | tensor of integer or boolean element types |

### Results

| Name     | Type                                       |
|----------|--------------------------------------------|
| `result` | tensor of integer or boolean element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// Bitwise operation with with integer tensors
// %lhs: [[1, 2], [3, 4]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.and"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[1, 2], [3, 0]]

// Logical operation with with boolean tensors
// %lhs: [[false, false], [true, true]]
// %rhs: [[false, true], [false, true]]
%result = "stablehlo.and"(%lhs, %rhs) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
// %result: [[false, false], [false, true]]
```

[Back to Ops](#index-of-ops)

## stablehlo.ceil

### Semantics

Performs element-wise ceil of `operand` tensor and produces a `result` tensor.
Implements the rounding to integral towards positive infinity operation from the
IEEE-754 specification.

### Operands

| Name      | Type                                   |
|-----------|----------------------------------------|
| `operand` | tensor of floating-point element types |

### Results

| Name     | Type                                   |
|----------|----------------------------------------|
| `result` | tensor of floating-point element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [-0.8166, -0.2530, 0.2530, 0.8166, 2.0]
%result = "stablehlo.ceil"(%operand) : (tensor<5xf32>) -> tensor<5xf32>
// %result: [-0.0, -0.0, 1.0, 1.0, 2.0]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_ceil.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.constant

### Semantics

Produces a `result` tensor from a constant `value`.

### Operands

| Name    | Type                            |
|---------|---------------------------------|
| `value` | constant of any supported types |

### Results

| Name     | Type                          |
|----------|-------------------------------|
| `result` | tensor of any supported types |

### Constraints

  * (C1) `value` and `result` have the same type.

### Examples

```mlir
%result = "stablehlo.constant"() {value = dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
// %result: [[0.0, 1.0], [2.0, 3.0]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_constant.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.cosine

### Semantics

Performs element-wise cosine operation on `operand` tensor and produces a
`result` tensor, implementing the `cos` operation from the IEEE-754
specification. Numeric precision is implementation-defined.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [
//            [0.0, 1.57079632],       // [0, pi/2]
//            [3.14159265, 4.71238898] // [pi, 3pi/2]
//           ]
%result = "stablehlo.cosine"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[1.0, 0.0], [-1.0, 0.0]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_cosine.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.divide

### Semantics

Performs element-wise division of dividend `lhs` and divisor `rhs` tensors and
produces a `result` tensor. For floating-point element types, it implements the
`division` operation from IEEE-754 specification. For integer element types, it
implements integer division truncating any fractional part. For n-bit integer
types, division overflow (division by zero or division of $-2^{n-1}$ with $-1$)
produces an implementation-defined value.

### Operands

| Name  | Type                                                       |
|-------|------------------------------------------------------------|
| `lhs` | tensor of integer, floating-point or complex element types |
| `rhs` | tensor of integer, floating-point or complex element types |

### Results

| Name     | Type                                                       |
|----------|------------------------------------------------------------|
| `result` | tensor of integer, floating-point or complex element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// %lhs: [17.1, -17.1, 17.1, -17.1]
// %rhs: [3.0, 3.0, -3.0, -3.0]
%result = "stablehlo.divide"(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// %result: [5.66666651, -5.66666651, -5.66666651, 5.66666651]

// %lhs: [17, -17, 17, -17]
// %rhs: [3, 3, -3, -3]
%result = "stablehlo.divide"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// %result: [5, -5, -5, 5]
```

[Back to Ops](#index-of-ops)

## stablehlo.exponential

### Semantics

Performs element-wise exponential operation on `operand` tensor and produces a
`result` tensor. For floating-point element types, it implements the `exp`
operation from the IEEE-754 specification. For complex element types, it
computes a complex exponential, with corner cases TBD. Numeric precision is
implementation-defined.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [[0.0, 1.0], [2.0, 3.0]]
%result = "stablehlo.exponential"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[1.0, 2.71828183], [7.38905610, 20.08553692]]

// %operand: (1.0, 2.0)
%result = "stablehlo.exponential"(%operand) : (tensor<complex<f32>>) -> tensor<complex<f32>>
// %result: (-1.13120438, 2.47172667)
```

[Back to Ops](#index-of-ops)

## stablehlo.floor

### Semantics

Performs element-wise floor of `operand` tensor and produces a `result` tensor.
Implements the rounding to integral towards negative infinity operation from the
IEEE-754 specification.

### Operands

| Name      | Type                                   |
|-----------|----------------------------------------|
| `operand` | tensor of floating-point element types |

### Results

| Name     | Type                                   |
|----------|----------------------------------------|
| `result` | tensor of floating-point element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [-0.8166, -0.2530, 0.2530, 0.8166, 2.0]
%result = "stablehlo.floor"(%operand) : (tensor<5xf32>) -> tensor<5xf32>
// %result: [-1.0, -1.0, 0.0, 0.0, 2.0]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_floor.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.log

### Semantics

Performs element-wise logarithm operation on `operand` tensor and produces a
`result` tensor. For floating-point element types, it implements the `log`
operation from the IEEE-754 specification. For complex element types, it
computes a complex logarithm, with corner cases TBD. Numeric precision is
implementation-defined.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [[1.0, 2.0], [3.0, 4.0]]
%result = "stablehlo.log"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[0.0, 0.69314718], [1.09861229, 1.38629436]]

// %operand: (1.0, 2.0)
%result = "stablehlo.log"(%operand) : (tensor<complex<f32>>) -> tensor<complex<f32>>
// %result: (0.80471896, 1.10714871)
```

[Back to Ops](#index-of-ops)

## stablehlo.logistic

### Semantics

Performs element-wise logistic (sigmoid) function on `operand` tensor and
produces a `result` tensor. For floating-point element types, it implements:
$$logistic(x) = division(1, addition(1, exp(-x)))$$
where `addition`, `division`, and `exp` are operations from IEEE-754
specification. For complex element types, it computes a complex logistic
function, with corner cases TBD. Numeric precision is implementation-defined.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [[0.0, 1.0], [2.0, 3.0]]
%result = "stablehlo.logistic"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[0.5, 0.73105858], [0.88079708, 0.95257413]]

// %operand: (1.0, 2.0)
%result = "stablehlo.logistic"(%operand) : (tensor<complex<f32>>) -> tensor<complex<f32>>
// %result: (1.02141536, 0.40343871)
```

[Back to Ops](#index-of-ops)

## stablehlo.maximum

### Semantics

Performs element-wise max operation on tensors `lhs` and `rhs` and produces a
`result` tensor. For floating-point element types, it implements the `maximum`
operation from the IEEE-754 specification. For complex element type, it performs
lexicographic comparison on the (real, imaginary) pairs with corner cases TBD.

### Operands

| Name  | Type                                                        |
|-------|-------------------------------------------------------------|
| `lhs` | tensor of integer, floating-point, or complex element types |
| `rhs` | tensor of integer, floating-point, or complex element types |

### Results

| Name     | Type                                                        |
|----------|-------------------------------------------------------------|
| `result` | tensor of integer, floating-point, or complex element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// %lhs: [[1, 2], [7, 8]]
// %rhs: [[5, 6], [3, 4]]
%result = "stablehlo.maximum"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[5, 6], [7, 8]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_maximum.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.minimum

### Semantics

Performs element-wise max operation on tensors `lhs` and `rhs` and produces a
`result` tensor. For floating-point element types, it implements the `minimum`
operation from the IEEE-754 specification. For complex element type, it performs
lexicographic comparison on the (real, imaginary) pairs with corner cases TBD.

### Operands

| Name  | Type                                                        |
|-------|-------------------------------------------------------------|
| `lhs` | tensor of integer, floating-point, or complex element types |
| `rhs` | tensor of integer, floating-point, or complex element types |

### Results

| Name     | Type                                                        |
|----------|-------------------------------------------------------------|
| `result` | tensor of integer, floating-point, or complex element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// %lhs: [[1, 2], [7, 8]]
// %rhs: [[5, 6], [3, 4]]
%result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[1, 2], [3, 4]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_minimum.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.negate

### Semantics

Performs element-wise negation of `operand` tensor and produces a `result`
tensor. For floating-point element types, it implements the `negate` operation
from the IEEE-754 specification. For signed integer types, it performs the
regular negation operation where the negation of $-2^{n-1}$ is implementation-
defined and one of the following:

  * Saturation to $2^{n-1}-1$
  * $-2^{n-1}$

For unsigned integer types, it bitcasts to the corresponding signed integer type,
performs the regular negation operation and bitcasts back to the original
unsigned integer type.

### Operands

| Name      | Type                                                        |
|-----------|-------------------------------------------------------------|
| `operand` | tensor of integer, floating-point, or complex element types |

### Results

| Name     | Type                                                        |
|----------|-------------------------------------------------------------|
| `result` | tensor of integer, floating-point, or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// Negation operation with integer Tensors
// %operand: [0, -2]
%result = "stablehlo.negate"(%operand) : (tensor<2xi32>) -> tensor<2xi32>
// %result: [0, 2]

// Negation operation with with complex tensors
// %operand: (2.5, 0.0)
%result = "stablehlo.negate"(%operand) : (tensor<1xcomplex<f32>>) -> tensor<1xcomplex<f32>>
// %result: [-2.5, -0.0]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_negate.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.not

### Semantics

Performs element-wise bitwise NOT of tensor `operand` of type integer and
produces a `result` tensor. For boolean tensors, it computes the logical NOT.

### Arguments

| Name      | Type                                       |
|-----------|--------------------------------------------|
| `operand` | tensor of integer or boolean element types |

### Results

| Name     | Type                                       |
|----------|--------------------------------------------|
| `result` | tensor of integer or boolean element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// Bitwise operation with with integer tensors
// %operand: [[1, 2], [3, 4]]
%result = "stablehlo.not"(%operand) : (tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[-2, -3], [-4, -5]]

// Bitwise operation with with boolean tensors
// %operand: [true, false]
%result = "stablehlo.not"(%operand) : (tensor<2xi1>) -> tensor<2xi1>
// %result: [false, true]
```

[Back to Ops](#index-of-ops)

## stablehlo.or

### Semantics

Performs element-wise bitwise OR of two tensors `lhs` and `rhs` of integer types
and produces a `result` tensor. For boolean tensors, it computes the logical
operation.

### Operands

| Name  | Type                                       |
|-------|--------------------------------------------|
| `lhs` | tensor of integer or boolean element types |
| `rhs` | tensor of integer or boolean element types |

### Results

| Name     | Type                                       |
|----------|--------------------------------------------|
| `result` | tensor of integer or boolean element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// Bitwise operation with with integer tensors
// %lhs: [[1, 2], [3, 4]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.or"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[5, 6], [7, 12]]

// Logical operation with with boolean tensors
// %lhs: [[false, false], [true, true]]
// %rhs: [[false, true], [false, true]]
%result = "stablehlo.or"(%lhs, %rhs) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
// %result: [[false, true], [true, true]]
```

[Back to Ops](#index-of-ops)

## stablehlo.remainder

### Semantics

Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors and
produces a `result` tensor. The sign of the result is taken from the dividend,
and the absolute value of the result is always less than the divisor's absolute
value. The remainder is calculated as `lhs - d * rhs`, where
`d = stablehlo.divide`. For floating-point element types, this is in contrast
with the `remainder` operation from IEEE-754 specification where `d` is an
integral value nearest to the exact value of `lhs/rhs` with ties to even. For
floating-point types, the corner cases are TBD. For n-bit integer, division
overflow (remainder by zero or remainder of $-2^{n-1}$ with $-1$) produces an
implementation-defined value.

### Operands

| Name  | Type                                                       |
|-------|------------------------------------------------------------|
| `lhs` | tensor of integer, floating-point or complex element types |
| `rhs` | tensor of integer, floating-point or complex element types |

### Results

| Name     | Type                                                       |
|----------|------------------------------------------------------------|
| `result` | tensor of integer, floating-point or complex element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// %lhs: [17.1, -17.1, 17.1, -17.1]
// %rhs: [3.0, 3.0, -3.0, -3.0]
%result = "stablehlo.remainder"(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// %result: [2.1, -2.1, 2.1, -2.1]

// %lhs: [17, -17, 17, -17]
// %rhs: [3, 3, -3, -3]
%result = "stablehlo.remainder"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// %result: [2, -2, 2, -2]
```

[Back to Ops](#index-of-ops)

## stablehlo.reshape

### Semantics

Performs reshape of `operand` tensor to a `result` tensor. Conceptually, it
amounts to keeping the same canonical representation but potentially changing
the shape, e.g. from `tensor<2x3xf32>` to `tensor<3x2xf32>` or `tensor<6xf32>`.

More formally, `result[i0, ..., iR-1] = operand[j0, ..., jR'-1]` where
`i` and `j` have the same position in the lexicographic ordering of the index
spaces of `result` and `operand`.

### Operands

| Name      | Type                          |
|-----------|-------------------------------|
| `operand` | tensor of any supported types |

### Results

| Name     | Type                          |
|----------|-------------------------------|
| `result` | tensor of any supported types |

### Constraints

  * (C1) `operand` and `result` have the same element type.
  * (C2) `operand` and `result` have the same number of elements.

### Examples

```mlir
// %operand: [[1, 2, 3], [4, 5, 6]]]
%result = "stablehlo.reshape"(%operand) : (tensor<2x3xi32>) -> tensor<3x2xi32>
// %result: [[1, 2], [3, 4], [5, 6]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_reshape.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.rsqrt

### Semantics

Performs element-wise reciprocal square root operation on `operand` tensor and
produces a `result` tensor, implementing the `rSqrt` operation from the IEEE-754
specification. Numeric precision is implementation-defined.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [[1.0, 4.0], [9.0, 25.0]]
%result = "stablehlo.rsqrt"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[1.0, 0.5], [0.33333343, 0.2]]

// %operand: [(1.0, 2.0)]
%result = "stablehlo.rsqrt"(%operand) : (tensor<complex<f32>>) -> tensor<complex<f32>>
// %result: [(0.56886448, -0.35157758)]
```

[Back to Ops](#index-of-ops)

## stablehlo.sine

### Semantics

Performs element-wise sine operation on `operand` tensor and produces a `result`
tensor, implementing the `sin` operation from the IEEE-754 specification.
Numeric precision is implementation-defined.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [
//            [0.0, 1.57079632],       // [0, pi/2]
//            [3.14159265, 4.71238898] // [pi, 3pi/2]
//           ]
%result = "stablehlo.sine"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[0.0, 1.0], [0.0, -1.0]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_sine.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.sqrt

### Semantics

Performs element-wise square root operation on `operand` tensor and produces a
`result` tensor, implementing the `squareRoot` operation from the IEEE-754
specification.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [[0.0, 1.0], [4.0, 9.0]]
%result = "stablehlo.sqrt"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[0.0, 1.0], [2.0, 3.0]]

// %operand: [(1.0, 2.0)]
%result = "stablehlo.sqrt"(%operand) : (tensor<complex<f32>>) -> tensor<complex<f32>>
// %result: [(1.27201965, 0.78615138)]
```

[Back to Ops](#index-of-ops)

## stablehlo.subtract

### Semantics

Performs element-wise subtraction of two tensors `lhs` and `rhs` and produces a
`result` tensor. For integer element types, if the element-wise difference has
an unsigned/signed overflow/underflow, the result is implementation-defined and
one of the following:

  * mathematical result modulo $2^n$, where n is the bit width of the result,
  for unsigned overflow/underflow. For signed integer overflow/underflow, wraps
  the result around the representable range $[-2^{n-1},\ 2^{n-1} - 1]$.
  * saturation to $2^{n-1} - 1$ (or $-2^{n-1}$) for signed overflow (or signed
  underflow) and saturation to $2^n - 1$ (or $0$) for unsigned overflow (or
  unsigned underflow).

For floating-point element types, it implements the `subtraction` operation from
the IEEE-754 specification.

### Operands

| Name  | Type                                                        |
|-------|-------------------------------------------------------------|
| `lhs` | tensor of integer, floating-point, or complex element types |
| `rhs` | tensor of integer, floating-point, or complex element types |

### Results

| Name     | Type                                                        |
|----------|-------------------------------------------------------------|
| `result` | tensor of integer, floating-point, or complex element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// %lhs: [[6, 8], [10, 12]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
// %result: [[1, 2], [3, 4]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_subtract.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.tanh

### Semantics

Performs element-wise tanh operation on `operand` tensor and produces a `result`
tensor, implementing the `tanh` operation from the IEEE-754 specification.
Numeric precision is implementation-defined.

### Operands

| Name      | Type                                              |
|-----------|---------------------------------------------------|
| `operand` | tensor of floating-point or complex element types |

### Results

| Name     | Type                                              |
|----------|---------------------------------------------------|
| `result` | tensor of floating-point or complex element types |

### Constraints

  * (C1) `operand` and `result` have the same type.

### Examples

```mlir
// %operand: [-1.0, 0.0, 1.0]
%result = "stablehlo.tanh"(%operand) : (tensor<3xf32>) -> tensor<3xf32>
// %result: [-0.76159416, 0.0, 0.76159416]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_tanh.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.transpose

### Semantics

Permutes the dimensions of `operand` tensor using `permutation` and produces a
`result` tensor. More formally, `result[i0, ..., iR-1] = operand[j0, ..., jR-1]`
where `i[d] = j[permutation[d]]`.

### Operands

| Name          | Type                               |
|---------------|------------------------------------|
| `operand`     | tensor of any supported types      |
| `permutation` | 1-dimensional array of type `si64` |

### Results

| Name     | Type                          |
|----------|-------------------------------|
| `result` | tensor of any supported types |

### Constraints

  * (C1) `operand` and `result` have the same element type.
  * (C2) `permutation` is a permutation of `[0, 1, ..., R-1]` where `R` is the
  rank of `operand`.
  * (C3) `result`'s shape is a permutation of `operand`'s shape.

### Examples

```mlir
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
%result = "stablehlo.transpose"(%operand) {
  permutation = dense<[2, 1, 0]> : tensor<3xi64>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// %result: [
//           [[1,7], [3,9], [5,11]],
//           [[2,8], [4,10], [6,12]]
//          ]
```

&nbsp;[More Examples](../stablehlo/tests/interpret_transpose.mlir)

[Back to Ops](#index-of-ops)

## stablehlo.xor

### Semantics

Performs element-wise bitwise XOR of two tensors `lhs` and `rhs` of integer
types and produces a `result` tensor. For boolean tensors, it computes the
logical operation.

### Operands

| Name  | Type                                       |
|-------|--------------------------------------------|
| `lhs` | tensor of integer or boolean element types |
| `rhs` | tensor of integer or boolean element types |

### Results

| Name     | Type                                       |
|----------|--------------------------------------------|
| `result` | tensor of integer or boolean element types |

### Constraints

  * (C1) `lhs`, `rhs` and `result` have the same type.

### Examples

```mlir
// Bitwise operation with with integer tensors
// %lhs: [[1, 2], [3, 4]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.xor"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[4, 4], [4, 12]]

// Logical operation with with boolean tensors
// %lhs: [[false, false], [true, true]]
// %rhs: [[false, true], [false, true]]
%result = "stablehlo.xor"(%lhs, %rhs) : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
// %result: [[false, true], [true, false]]
```

[Back to Ops](#index-of-ops)
