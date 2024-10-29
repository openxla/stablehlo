# StableHLO Specification

StableHLO is an operation set for high-level operations (HLO) in machine
learning (ML) models. StableHLO works as a portability layer between different
ML frameworks and ML compilers: ML frameworks that produce StableHLO programs
are compatible with ML compilers that consume StableHLO programs.

Our goal is to simplify and accelerate ML development by creating more
interoperability between various ML frameworks (such as TensorFlow, JAX and
PyTorch) and ML compilers (such as XLA and IREE). Towards that end, this
document provides a specification for the StableHLO programming language.

This specification contains three major sections. First, the
[Programs](#programs) section describes the structure of StableHLO programs
which consist of StableHLO functions which themselves consist of StableHLO ops.
Within that structure, the [Ops](#ops) section specifies the semantics of
individual ops. The [Execution](#execution) section provides semantics for all
these ops executing together within a program. Finally, the
[Notation](#notation) section discusses the notation used throughout the
specification.

To view the spec from a previous release of StableHLO, open the repo at the
[tagged release](https://github.com/openxla/stablehlo/tags) of interest.
For example, the [StableHLO v0.19.0 Spec](https://github.com/openxla/stablehlo/blob/v0.19.0/docs/spec.md).
To view changes that occurred at each minor version bump of StableHLO, refer to
the version log in [VhloDialect.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloDialect.td).

## Programs

```ebnf
Program ::= {Func}
```

**StableHLO programs** consist of an arbitrary number of StableHLO functions.
Below is an example program with a function `@main` which has 3 inputs
(`%image`, `%weights` and `%bias`) and 1 output. The body of the function
has 6 ops.

```mlir
func.func @main(
  %image: tensor<28x28xf32>,
  %weights: tensor<784x10xf32>,
  %bias: tensor<1x10xf32>
) -> tensor<1x10xf32> {
  %0 = "stablehlo.reshape"(%image) : (tensor<28x28xf32>) -> tensor<1x784xf32>
  %1 = "stablehlo.dot"(%0, %weights) : (tensor<1x784xf32>, tensor<784x10xf32>) -> tensor<1x10xf32>
  %2 = "stablehlo.add"(%1, %bias) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  %3 = "stablehlo.constant"() {value = dense<0.0> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  %4 = "stablehlo.maximum"(%2, %3) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  "func.return"(%4): (tensor<1x10xf32>) -> ()
}
```

### Functions

```ebnf
Func        ::= 'func' '.' 'func' FuncId FuncInputs FuncOutputs '{' FuncBody '}'
FuncInputs  ::= '(' [FuncInput {',' FuncInput}] `)`
FuncInput   ::= ValueId ':' ValueType
FuncOutputs ::= ['->' FuncOutput, {',' FuncOutput}]
FuncOutput  ::= ValueType
FuncBody    ::= {Op}
```

**StableHLO functions** (which are also called **named functions**) have
an identifier, inputs/outputs and a body. In the future, we are planning to
introduce additional metadata for functions to achieve better compatibility
with HLO ([#425](https://github.com/openxla/stablehlo/issues/425),
[#626](https://github.com/openxla/stablehlo/issues/626),
[#740](https://github.com/openxla/stablehlo/issues/740),
[#744](https://github.com/openxla/stablehlo/issues/744)).

### Identifiers

```ebnf
FuncId  ::= '@' letter {letter | digit}
ValueId ::= '%' digit {digit}
          | '%' letter {letter | digit}
letter  ::= 'a' | ... | 'z' | 'A' | ... | 'Z' | '_'
digit   ::= '0' | ... | '9'
```

**StableHLO identifiers** are similar to identifiers in many programming
languages, with two peculiarities: 1) all identifiers have sigils which
distinguish different kinds of identifiers, 2) value identifiers can be
completely numeric to simplify generation of StableHLO programs.

### Types

```ebnf
Type         ::= ValueType | NonValueType
ValueType    ::= TensorType | QuantizedTensorType | TokenType | TupleType
NonValueType ::= TensorElementType | QuantizedTensorElementType | FunctionType | StringType
```

**StableHLO types** are categorized into **value types** (which are also called
**first-class types**) which represent StableHLO values and **non-value types**
which describe other program elements. StableHLO types are similar to types in
many programming languages, with the main peculiarity being StableHLO's
domain-specific nature which results in some unusual outcomes (e.g. scalar types
are not value types).

```ebnf
TensorType ::= 'tensor' '<' Shape TensorElementType '>'
Shape ::= {DimensionSize 'x'}
DimensionSize ::= digit {digit} | '?'
```

**Tensor types** represent tensors, i.e. multidimensional arrays. They have a
**shape** and an **element type**, where a shape represents non-negative or
unknown **dimension sizes** in the ascending order of the corresponding
**dimensions** (which are also called **axes**) numbered from `0` to `R-1`. The
number of dimensions `R` is called **rank**. For example, `tensor<2x3xf32>` is
a tensor type with shape `2x3` and element type `f32`. It has two dimensions
(or, in other words, two axes) - 0th dimension and 1st dimension - whose sizes
are 2 and 3. Its rank is 2.

Shapes can be partially or completely unknown (dynamic), e.g. `tensor<?x2xf64>`
is partially unknown and `tensor<?x?xf64>` is completely unknown. Dynamic
dimension sizes are represented using a `?`. Shapes cannot be unranked.

In the future, we are planning to explore extending tensor types beyond
dimension sizes and element types, for example, to include layouts
([#629](https://github.com/openxla/stablehlo/issues/629)) and sparsity
([#1078](https://github.com/openxla/stablehlo/issues/1078)).

```ebnf
QuantizedTensorType ::= 'tensor' '<' Shape QuantizedTensorElementType '>'
QuantizedTensorElementType ::= '!quant.uniform' '<'
                  QuantizationStorageType
                  ['<' QuantizationStorageMin ':' QuantizationStorageMax '>']
                  ':' QuantizationExpressedType
                  [':' QuantizationDimension]
                  ',' QuantizationParameters '>'
QuantizationStorageType ::= IntegerType
QuantizationStorageMin ::= IntegerLiteral
QuantizationStorageMax ::= IntegerLiteral
QuantizationExpressedType ::= FloatType
QuantizationDimension ::= IntegerLiteral
QuantizationParameters ::= QuantizationParameter
                         | '{' QuantizationParameter {',' QuantizationParameter} '}'
QuantizationParameter ::= QuantizationScale [':' QuantizationZeroPoint]
QuantizationScale ::= FloatLiteral
QuantizationZeroPoint ::= IntegerLiteral
```

| Name                     | Type                                        | Constraints                 |
|--------------------------|---------------------------------------------|-----------------------------|
| `storage_type`           | integer type                                | (C1-C3), (C8)               |
| `storage_min`            | integer constant                            | (C1), (C3), (C7)            |
| `storage_max`            | integer constant                            | (C2), (C3), (C7)            |
| `expressed_type`         | floating-point type                         | (C4)                        |
| `quantization_dimension` | optional integer constant                   | (C10-C12)                   |
| `scales`                 | variadic number of floating-point constants | (C4-C6), (C9), (C10), (C13) |
| `zero_points`            | variadic number of integer constants        | (C7-C9)                     |

**Quantized element types** represent integer values of a **storage type** in
the range from `storage_min` to `storage_max` (inclusive) that correspond to
floating-point values of an **expressed type**. For a given integer value `i`,
the corresponding floating-point value `f` can be computed as
`f = (i - zero_point) * scale`, where `scale` and `zero_point` are called
**quantization parameters**. The `storage_min` and `storage_max` are optional
in the grammar, but have default values of `min_value(storage_type)` and
`max_value(storage_type)` respectively. Quantized element types have the
following constraints:

* (C1) `type(storage_min) = storage_type`.
* (C2) `type(storage_max) = storage_type`.
* (C3) `min_value(storage_type) <= storage_min < storage_max <= max_value(storage_type)`.
* (C4) `type(scales...) = expressed_type`.
* (C5) `0 < scales`.
* (C6) `is_finite(scales...)`.
* (C7) `storage_min <= zero_points <= storage_max`.
* (C8) `type(zero_points...) = storage_type`.
* (C9) `size(scales) = size(zero_points)`.
* (C10) If `is_empty(quantization_dimension)`, then `size(scales) = 1`.
* (C11) `0 <= quantization_dimension`.

At the moment, `QuantizationScale` is a floating-point constant, but there is
strong interest in integer-based scales, represented with multipliers and
shifts. We are planning to explore this in the near future
([#1404](https://github.com/openxla/stablehlo/issues/1404)).

There is an ongoing discussion on the semantics of `QuantizationZeroPoint`,
including the type, the values and whether there can be just one or
potentially multiple zero points in a quantized tensor type. Based on the
results of this discussion, the specification around zero points may change
in the future ([#1405](https://github.com/openxla/stablehlo/issues/1405)).

Another ongoing discussion involves the semantics of `QuantizationStorageMin`
and `QuantizationStorageMax` to determine whether any constraints should be
imposed on these values and on the values of quantized tensors
([#1406](https://github.com/openxla/stablehlo/issues/1406)).

Finally, we are planning to explore representing unknown scales and zero
points, similarly to how we are planning to explore representing unknown
dimension sizes ([#1407](https://github.com/openxla/stablehlo/issues/1407)).

**Quantized tensor types** represent tensors with quantized elements. These
tensors are exactly the same as regular tensors, except that their elements
have quantized element types, instead of regular element types.

In quantized tensors, quantization can be **per-tensor**, meaning, having
one `scale` and `zero_point` for the entire tensor or can be **per-axis**,
meaning, having multiple `scales` and `zero_points`, one pair per slice of
a particular dimension `quantization_dimension`. More formally, in a tensor `t`
with per-axis quantization, there are `dim(t, quantization_dimension)` slices
of the `quantization_dimension`: `t[:, ..., 0, ..., :], t[:, ..., 1, ..., :]`,
etc. All elements in the `i`th slice use `scales[i]` and `zero_points[i]` as
their quantization parameters. Quantized tensor types have the following
constraints:

* For per-tensor quantization:
  * No additional constraints.
* For per-axis quantization:
  * (C12) `quantization_dimension < rank(self)`.
  * (C13) `dim(self, quantization_dimension) = size(scales)`.

```ebnf
TokenType ::= 'token'
```

**Token types** represent tokens, i.e. opaque values produced and consumed
by some operations. Tokens are used for imposing execution order on operations
as described in the [Execution](#execution) section.

```ebnf
TupleType ::= 'tuple' '<' TupleElementTypes '>'
TupleElementTypes ::= [ValueType {',' ValueType}]
```

**Tuple types** represent tuples, i.e. heterogeneous lists. Tuples are a legacy
feature which only exists for compatibility with HLO. In HLO, tuples are
used to represent variadic inputs and outputs. In StableHLO, variadic inputs and
outputs are supported natively, and the only use of tuples in StableHLO is to
comprehensively represent HLO ABI where e.g. `T`, `tuple<T>` and
`tuple<tuple<T>>` may be materially different depending on a particular
implementation. In the future, we are planning to make changes to HLO ABI
which may allow us to remove tuple types from StableHLO
([#598](https://github.com/openxla/stablehlo/issues/598)).

```ebnf
TensorElementType ::= BooleanType | IntegerType | FloatType | ComplexType
BooleanType ::= 'i1'
IntegerType ::= SignedIntegerType | UnsignedIntegerType
SignedIntegerType ::= 'si2' | 'si4' | 'si8' | 'si16' | 'si32' | 'si64'
UnsignedIntegerType ::= 'ui2' | 'ui4' | 'ui8' | 'ui16' | 'ui32' | 'ui64'
FloatType ::= 'f4E2M1FN' | 'f6E2M3FN' | 'f6E3M2FN' | 'f8E3M4' | 'f8E4M3'
            | 'f8E4M3FN' | 'f8E4M3FNUZ' | 'f8E4M3B11FNUZ' | 'f8E5M2'
            | 'f8E5M2FNUZ' | 'f8E8M0FNU' | 'bf16' | 'f16' | 'f32' | 'f64'
TensorFloat32 ::= 'tf32'
ComplexType ::= 'complex' '<' ComplexElementType '>'
ComplexElementType ::= 'f32' | 'f64'
```

**Element types** represent elements of tensor types. Unlike in many programming
languages, these types are not first class in StableHLO. This means that
StableHLO programs cannot directly represent values of these types (as a result,
it is idiomatic to represent scalar values of type `T` with 0-dimensional tensor
values of type `tensor<T>`).

* **Boolean type** represents boolean values `true` and `false`.
* **Integer types** can be either signed (`si`) or unsigned (`ui`) and have
  one of the supported bit widths (`2`, `4`, `8`, `16`, `32` or `64`).
  Signed `siN` types represent integer values from `-2^(N-1)` to `2^(N-1)-1`
  inclusive, and unsigned `uiN` types represent integer values from `0` to
  `2^N-1` inclusive.
* **Floating-point types** can be one of the following:
  * `f8E3M4`, `f8E4M3` and `f8E5M2` 8-bit floating point numbers following
    IEEE-754 conventions.
  * `f8E4M3FN` and `f8E5M2` types corresponding to respectively the
    `E4M3` and `E5M2` encodings of the FP8 format described in
    [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433).
  * `f8E4M3FNUZ` and `f8E5M2FNUZ` types corresponding to the `E4M3` and `E5M2`
    encodings of the FP8 formats described in
    [8-bit Numerical Formats for Deep Neural Networks](https://arxiv.org/abs/2206.02915).
  * `f8E4M3B11FNUZ` type corresponding to the `E4M3` encoding of the FP8 formats
    described in
    [Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf).
  * `bf16` type corresponding to the `bfloat16` format described in
    [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus).
  * `f16`, `f32` and `f64` types corresponding to respectively
    `binary16` ("half precision"), `binary32` ("single precision") and
    `binary64` ("double precision") formats described in
    [the IEEE 754 standard](https://ieeexplore.ieee.org/document/8766229).
  * `tf32` type corresponds to the [TensorFloat32 format](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)
    and has limited support in StableHLO.
  * `f4E2M1FN`, `f6E2M3FN`, `f6E3M2FN` and `f8E8M0FNU` MX (microscaling) types
    described in
    [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
* **Complex types** represent complex values that have a **real part**
  and an **imaginary part** of the same **element type**. Supported complex
  types are `complex<f32>` (both parts are of type `f32`) and `complex<f64>`
  (both parts are of type `f64`).

```ebnf
FunctionType ::= '(' InputTypes ')' '->' '(' OutputTypes ')'
InputTypes ::= [ValueType {',' ValueType}]
OutputTypes ::= [ValueType {',' ValueType}]
```

**Function types** represent both named and anonymous functions. They have
input types (the list of types on the left-hand side of `->`) and output types
(the list of types on the right-hand side of `->`). In many programming
languages, function types are first class, but not in StableHLO.

```ebnf
StringType ::= 'string'
```

**String type** represents sequences of bytes. Unlike in many programming
languages, string type is not first class in StableHLO and is only used to
specify static metadata for program elements.

### Operations

**StableHLO operations** (which are also called **ops**) represent a closed set
of high-level operations in machine learning models. As discussed above,
StableHLO syntax is heavily inspired by MLIR, which is not necessarily the most
ergonomic alternative, but is arguably the best fit for StableHLO's goal of
creating more interoperability between ML frameworks and ML compilers.

```ebnf
Op            ::= [OpOutputs] OpName OpInputs ':' OpSignature
OpName        ::= '"' 'stablehlo' '.' OpMnemonic '"'
OpMnemonic    ::= 'abs' | 'add' | ...
```

**StableHLO operations** (which are also called **ops**) have a name,
inputs/outputs and a signature. The name consists of the `stablehlo.` prefix and
a **mnemonic** which uniquely identifies one of the supported ops. See below for
a comprehensive list of all supported ops.

```ebnf
OpInputs        ::= OpInputValues OpInputFuncs OpInputAttrs
OpInputValues   ::= '(' [OpInputValue {',' OpInputValue}] ')'
OpInputValue    ::= ValueId
OpInputFuncs    ::= ['(' OpInputFunc {',' OpInputFunc} ')']
OpInputAttrs    ::= ['{' OpInputAttr {',' OpInputAttr} '}']
OpOutputs       ::= [OpOutput {',' OpOutput} '=']
OpOutput        ::= ValueId
```

Ops consume **inputs** and produce **outputs**. Inputs are categorized into
input values (computed during execution), input functions (provided
statically, because in StableHLO functions are not first-class values) and
input attributes (also provided statically). The kind of inputs and outputs
consumed and produced by an op depends on its mnemonic. For example, the `add`
op consumes 2 input values and produces 1 output value. In comparison, the
`select_and_scatter` op consumes 3 input values, 2 input functions and
3 input attributes.

```ebnf
OpInputFunc ::= '{' Unused FuncInputs ':' FuncBody '}'
Unused      ::= '^' digit {digit}
              | '^' letter {letter | digit}
```

**Input functions** (which are also called **anonymous functions**) are very
similar to named functions except that: 1) they don't have an identifier (hence
the name "anonymous"), 2) they don't declare output types (output types are
inferred from the `return` op within the function).

The syntax for input functions includes a currently unused part (see the
`Unused` production above) which is there for compatibility with MLIR. In MLIR,
there is a more general concept of "regions" which can have multiple "blocks"
of ops connected together via jump ops. These blocks have ids which correspond
to the `Unused` production, so that they can be distinguished from each other.
StableHLO doesn't have jump ops, so the corresponding part of MLIR syntax is
unused (but is still there).

```ebnf
OpInputAttr      ::= OpInputAttrName '=' OpInputAttrValue
OpInputAttrName  ::= letter {letter | digit}
OpInputAttrValue ::= Constant
```

**Input attributes** have a name and a value which is one of the supported
constants. They are the primary way to specify static metadata for program
elements. For example, the `concatenate` op uses the attribute `dimension` to
specify the dimension along which its input values are concatenated. Similarly,
the `slice` op uses multiple attributes like `start_indices` and `limit_indices`
to specify the bounds that are used to slice the input value.

At the moment, StableHLO programs in the wild sometimes contain attributes
which are not described in this document. In the future, we are planning to
either absorb these attributes into the StableHLO opset or prohibit them from
appearing in StableHLO programs. In the meanwhile, here is the list of these
attributes:

* `layout` ([#629](https://github.com/openxla/stablehlo/issues/629)).
* `mhlo.frontend_attributes`
  ([#628](https://github.com/openxla/stablehlo/issues/628)).
* `mhlo.sharding` ([#619](https://github.com/openxla/stablehlo/issues/619)).
* `output_operand_aliases`
  ([#740](https://github.com/openxla/stablehlo/issues/740)).
* Location metadata ([#594](https://github.com/openxla/stablehlo/issues/594)).

```ebnf
OpSignature ::= '(' [ValueType {',' ValueType}] ')' '->' '(' [ValueType {',' ValueType}] ')'
```

**Op signature** consists of the types of all input values (the list of types on
the left-hand side of `->`) and the types of all output values (the list of
types on the right-hand side of `->`). Strictly speaking, input types are
redundant, and output types are almost always redundant as well (because for
most StableHLO ops, output types can be inferred from inputs). Nonetheless, op
signature is deliberately part of StableHLO syntax for compatibility with MLIR.

Below is an example op whose mnemonic is `select_and_scatter`. It consumes 3
input values (`%operand`, `%source` and `%init_value`), 2 input functions
and 3 input attributes (`window_dimensions`, `window_strides` and `padding`).
Note how the signature of the op only includes the types of its input values
(but not the types of input functions and attributes which are provided inline).

```mlir
%result = "stablehlo.select_and_scatter"(%operand, %source, %init_value) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %0 = "stablehlo.compare"(%arg0, %arg1) {
      comparison_direction = #stablehlo<comparison_direction GE>
    } : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%0) : (tensor<i1>) -> ()
}, {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%0) : (tensor<i32>) -> ()
}) {
  window_dimensions = dense<[3, 1]> : tensor<2xi64>,
  window_strides = dense<[2, 1]> : tensor<2xi64>,
  padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
} : (tensor<4x2xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x2xi32>
```

### Constants

```ebnf
Constant ::= BooleanConstant
           | IntegerConstant
           | FloatConstant
           | ComplexConstant
           | TensorConstant
           | QuantizedTensorConstant
           | StringConstant
           | EnumConstant
```

**StableHLO constants** have a literal and a type which together represent
a StableHLO value. Generally, the type is part of the constant syntax, except
when it's unambiguous (e.g. a boolean constant unambiguously has type `i1`,
whereas an integer constant can have multiple possible types).

```ebnf
BooleanConstant ::= BooleanLiteral
BooleanLiteral  ::= 'true' | 'false'
```

**Boolean constants** represent boolean values `true` and `false`. Boolean
constants have type `i1`.

```ebnf
IntegerConstant   ::= IntegerLiteral ':' IntegerType
IntegerLiteral    ::= ['-' | '+'] DecimalDigits
                    | ['-' | '+'] '0x' HexadecimalDigits
DecimalDigits     ::= decimalDigit {decimalDigit}
HexadecimalDigits ::= hexadecimalDigit {hexadecimalDigit}
decimalDigit      ::= '0' | ... | '9'
hexadecimalDigit  ::= decimalDigit | 'a' | ... | 'f' | 'A' | ... | 'F'
```

**Integer constants** represent integer values via strings that use decimal or
hexadecimal notation. Other bases, e.g. binary or octal, are not supported.
Integer constants have the following constraints:

* (C1) `is_wellformed(integer_literal, integer_type)`.

```ebnf
FloatConstant  ::= FloatLiteral ':' FloatType
FloatLiteral   ::= SignPart IntegerPart FractionalPart ScientificPart
                 | '0x' [HexadecimalDigits]
SignPart       ::= ['-' | '+']
IntegerPart    ::= DecimalDigits
FractionalPart ::= ['.' [DecimalDigits]]
ScientificPart ::= [('e' | 'E') ['-' | '+'] DecimalDigits]
```

**Floating-point constants** represent floating-point values via strings that
use decimal or scientific notation. Additionally, hexadecimal notation can be
used to directly specify the underlying bits in the floating-point format of
the corresponding type. Floating-point constants have the following constraints:

* (C1) If non-hexadecimal notation is used,
  `is_wellformed(float_literal, float_type)`.
* (C2) If hexadecimal notation is used,
  `size(hexadecimal_digits) = num_bits(float_type) / 4`.

```ebnf
ComplexConstant ::= ComplexLiteral ':' ComplexType
ComplexLiteral  ::= '(' RealPart ',' ImaginaryPart ')'
RealPart        ::= FloatLiteral
ImaginaryPart   ::= FloatLiteral
```

**Complex constants** represent complex values using lists of a real part
(comes first) and an imaginary part (comes second). For example,
`(1.0, 0.0) : complex<f32>` represents `1.0 + 0.0i`, and
`(0.0, 1.0) : complex<f32>` represents `0.0 + 1.0i`. The order in which these
parts are then stored in memory is implementation-defined. Complex constants
have the following constraints:

* (C1) `is_wellformed(real_part, complex_element_type(complex_type))`.
* (C2) `is_wellformed(imaginary_part, complex_element_type(complex_type))`.

```ebnf
TensorConstant ::= TensorLiteral ':' TensorType
TensorLiteral  ::= 'dense' '<' (DenseLiteral | ElementLiteral) '>'
DenseLiteral   ::= DenseDimension | DenseElements
DenseDimension ::= '[' [DenseLiteral {',' DenseLiteral}] ']'
DenseElements  ::= [ElementLiteral {',' ElementLiteral}]
ElementLiteral ::= BooleanLiteral | IntegerLiteral | FloatLiteral | ComplexLiteral
```

**Tensor constants** represent tensor values using nested lists specified via
NumPy notation. For example, `dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>`
represents a tensor value with the following mapping from indices to elements:
`{0, 0} => 1`, `{0, 1} => 2`, `{0, 2} => 3`, `{1, 0} => 4`, `{1, 1} => 5`,
`{1, 2} => 6`. The order in which these elements are then stored in memory is
implementation-defined. Tensor constants have the following constraints:

* (C1) `has_syntax(tensor_literal, element_type(tensor_type))`, where:
  * `has_syntax(element_literal: Syntax, element_type: Type) =
    is_wellformed(element_literal, type)`.
  * `has_syntax(tensor_literal: List, element_type: Type) =
    has_syntax(tensor_literal..., element_type)`.
* (C2) `has_shape(tensor_literal, shape(tensor_type))`, where:
  * `has_shape(element_literal: Syntax, []) = true`.
  * `has_shape(tensor_literal: List, shape: List) =
    size(tensor_literal) = shape[0] and
    has_shape(tensor_literal..., shape[1:])`.
  * otherwise, `false`.

```ebnf
QuantizedTensorConstant ::= QuantizedTensorLiteral ':' QuantizedTensorType
QuantizedTensorLiteral  ::= 'dense' '<' (DenseLiteral | ElementLiteral) '>'
```

**Quantized tensor constants** represent quantized tensor values using the same
notation as tensor constants, with elements specified as constants of their
storage type. Quantized tensor constants have the following constraints:

* (C1) `has_syntax(quantized_tensor_literal, storage_type(quantized_tensor_type))`.
* (C2) `has_shape(quantized_tensor_literal, shape(quantized_tensor_type))`.

```ebnf
StringConstant  ::= StringLiteral
StringLiteral   ::= '"' {stringCharacter | escapeSequence} '"'
stringCharacter ::= all ASCII characters except '\00', '\01', ... '\1f' and '"'
escapeSequence  ::= '\' ('"' | '\' | 'n' | 't' | (hexadecimalDigit hexadecimalDigit))
```

**String literals** consist of bytes specified using ASCII characters and
escape sequences. They are encoding-agnostic, so the interpretation of these
bytes is implementation-defined. String literals have type `string`.

## Ops

### abs

#### Semantics

Performs element-wise abs operation on `operand` tensor and produces a `result`
tensor. Depending on the element type, does the following:

* For signed integers: integer modulus.
* For floats: `abs` from IEEE-754.
* For complex numbers: complex modulus.
* For quantized types: `dequantize_op_quantize(abs, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                                     | Constraints |
|-------|-----------|------------------------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of signed integer, floating-point, or complex type or per-tensor quantized tensor | (C1-C2)     |

#### Outputs

| Name     | Type                                                                           | Constraints |
|----------|--------------------------------------------------------------------------------|-------------|
| `result` | tensor of signed integer or floating-point type or per-tensor quantized tensor | (C1-C2)     |

#### Constraints

* (C1) `shape(result) = shape(operand)`.
* (C2) `baseline_element_type(result)` is defined as:
  * `complex_element_type(element_type(operand))` if `is_complex(operand)`.
  * `baseline_element_type(operand)` otherwise.

#### Examples

```mlir
// %operand: [-2, 0, 2]
%result = "stablehlo.abs"(%operand) : (tensor<3xi32>) -> tensor<3xi32>
// %result: [2, 0, 2]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/abs.mlir)

### add

#### Semantics

Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
`result` tensor. Depending on the element type, does the following:

* For booleans: logical OR.
* For integers: integer addition.
* For floats: `addition` from IEEE-754.
* For complex numbers: complex addition.
* For quantized types: `dequantize_op_quantize(add, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                       | Constraints   |
|-------|-------|----------------------------|---------------|
| (I1)  | `lhs` | tensor or quantized tensor | (C1-C6)       |
| (I2)  | `rhs` | tensor or quantized tensor | (C1-C5), (C7) |

#### Outputs

| Name     | Type                       | Constraints |
|----------|----------------------------|-------------|
| `result` | tensor or quantized tensor | (C1-C7)     |

#### Constraints

* If the operation uses non-quantized tensors:
  * (C1) `type(lhs) = type(rhs) = type(result)`.
* If the operation uses quantized tensors:
  * (C2) `is_quantized(lhs) and is_quantized(rhs) and is_quantized(result)`.
  * (C3) `storage_type(lhs) = storage_type(rhs) = storage_type(result)`.
  * (C4) `expressed_type(lhs) = expressed_type(rhs) = expressed_type(result)`.
  * (C5) `(is_per_axis_quantized(lhs) or is_per_axis_quantized(rhs)) =
    is_per_axis_quantized(result)`.
  * (C6) If `is_per_axis_quantized(lhs)`, then `quantization_dimension(lhs) =
    quantization_dimension(result)`.
  * (C7) If `is_per_axis_quantized(rhs)`, then `quantization_dimension(rhs) =
    quantization_dimension(result)`.

#### Examples

```mlir
// %lhs: [[1, 2], [3, 4]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[6, 8], [10, 12]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/add.mlir)

### after_all

#### Semantics

Ensures that the operations producing the `inputs` are executed before any
operations that depend on `result`. Execution of this operation does nothing,
it only exists to establish data dependencies from `result` to `inputs`.

#### Inputs

| Label | Name     | Type                       |
|-------|----------|----------------------------|
| (I1)  | `inputs` | variadic number of `token` |

#### Outputs

| Name     | Type    |
|----------|---------|
| `result` | `token` |

#### Examples

```mlir
// %input0: !stablehlo.token
// %input1: !stablehlo.token
%result = "stablehlo.after_all"(%input0, %input1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/after_all.mlir)

### all_gather

#### Semantics

Within each process group in the StableHLO process grid, concatenates the values
of the `operands` tensors from each process along `all_gather_dim` and produces
`results` tensors.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)`
  if `channel_id <= 0 and use_global_device_ids = false`.
* `cross_replica_and_partition(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = false`.
* `flattened_ids(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = true`.

Afterwards, within each `process_group`:

* `operands...@receiver = [operand@sender for sender in process_group]` for all
  `receiver` in `process_group`.
* `results...@process = concatenate(operands...@process, all_gather_dim)` for all
  `process` in `process_group`.

#### Inputs

| Label | Name                    | Type                                                        | Constraints |
|-------|-------------------------|-------------------------------------------------------------|-------------|
| (I1)  | `operands`              | variadic number of tensors or per-tensor quantized tensors  | (C1), (C6)  |
| (I2)  | `all_gather_dim`        | constant of type `si64`                                     | (C1), (C6)  |
| (I3)  | `replica_groups`        | 2-dimensional tensor constant of type `si64`                | (C2-C4)     |
| (I4)  | `channel_id`            | constant of type `si64`                                     | (C5)        |
| (I5)  | `use_global_device_ids` | constant of type `i1`                                       | (C5)        |

#### Outputs

| Name      | Type                                                       | Constraints |
|-----------|------------------------------------------------------------|-------------|
| `results` | variadic number of tensors or per-tensor quantized tensors | (C6)        |

#### Constraints

* (C1) `0 <= all_gather_dim < rank(operands...)`.
* (C2) `is_unique(replica_groups)`.
* (C3) `size(replica_groups)` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_replicas` if `cross_replica_and_partition` is used.
  * `num_processes` if `flattened_ids` is used.
* (C4) `0 <= replica_groups < size(replica_groups)`.
* (C5) If `use_global_device_ids = true`, then `channel_id > 0`.
* (C6) `type(results...) = type(operands...)` except:
  * `dim(results..., all_gather_dim) =
    dim(operands..., all_gather_dim) * dim(process_groups, 1)`.

#### Examples

```mlir
// num_replicas: 2
// num_partitions: 1
// %operand0@(0, 0): [[1, 2], [3, 4]]
// %operand0@(1, 0): [[5, 6], [7, 8]]
// %operand1@(0, 0): [[11, 12], [13, 14]]
// %operand1@(1, 0): [[15, 16], [17, 18]]
%result:2 = "stablehlo.all_gather"(%operand0, %operand1) {
  all_gather_dim = 1 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  // channel_id = 0
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  // use_global_device_ids = false
} : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
// %result0@(0, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
// %result0@(1, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
// %result1@(0, 0): [[11, 12, 15, 16], [13, 14, 17, 18]]
// %result1@(1, 0): [[11, 12, 15, 16], [13, 14, 17, 18]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/all_gather.mlir)

### all_reduce

#### Semantics

Within each process group in the StableHLO process grid, applies a reduction
function `computation` to the values of the `operands` tensors from each process
and produces `results` tensors.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)`
  if `channel_id <= 0 and use_global_device_ids = false`.
* `cross_replica_and_partition(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = false`.
* `flattened_ids(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = true`.

Afterwards, within each `process_group`:

* `results...@process[result_index] = exec(schedule)` for some binary tree
  `schedule` where:
  * `exec(node)` = `computation(exec(node.left), exec(node.right))`.
  * `exec(leaf)` = `leaf.value`.
* `schedule` is an implementation-defined binary tree whose in-order
  traversal is `to_destination_type(operands...@process_group...[result_index],
  type(func_inputs(computation)[0]))`.

#### Inputs

| Label | Name                    | Type                                                             | Constraints |
|-------|-------------------------|------------------------------------------------------------------|-------------|
| (I1)  | `operands`              | variadic number of tensors or per-tensor quantized tensors       | (C5), (C6)  |
| (I2)  | `replica_groups`        | variadic number of 1-dimensional tensor constants of type `si64` | (C1-C3)     |
| (I3)  | `channel_id`            | constant of type `si64`                                          | (C4)        |
| (I4)  | `use_global_device_ids` | constant of type `i1`                                            | (C4)        |
| (I5)  | `computation`           | function                                                         | (C5)        |

#### Outputs

| Name      | Type                                                        | Constraints |
|-----------|-------------------------------------------------------------|-------------|
| `results` | variadic number of tensors or per-tensor quantized tensors  | (C6-C7)     |

#### Constraints

* (C1) `is_unique(replica_groups)`.
* (C2) `size(replica_groups)` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_replicas` if `cross_replica_and_partition` is used.
  * `num_processes` if `flattened_ids` is used.
* (C3) `0 <= replica_groups < size(replica_groups)`.
* (C4) If `use_global_device_ids = true`, then `channel_id > 0`.
* (C5) `computation` has type `(tensor<E>, tensor<E>) -> (tensor<E>)` where
       `is_promotable(element_type(operand), E)`.
* (C6) `shape(results...) = shape(operands...)`.
* (C7) `element_type(results...) = E`.

#### Examples

```mlir
// num_replicas: 2
// num_partitions: 1
// %operand0@(0, 0): [1, 2, 3, 4]
// %operand0@(1, 0): [5, 6, 7, 8]
// %operand1@(0, 0): [9, 10, 11, 12]
// %operand1@(1, 0): [13, 14, 15, 16]
%result:2 = "stablehlo.all_reduce"(%operand0, %operand0) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  // channel_id = 0
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  // use_global_device_ids = false
} : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
// %result0@(0, 0): [6, 8, 10, 12]
// %result0@(1, 0): [6, 8, 10, 12]
// %result1@(0, 0): [22, 24, 26, 28]
// %result1@(1, 0): [22, 24, 26, 28]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/all_reduce.mlir)

### all_to_all

#### Semantics

![all_to_all](images/spec/all_to_all.svg)

Within each process group in the StableHLO process grid, splits the values of
the `operands` tensors along `split_dimension` into parts, scatters the split
parts between the processes, concatenates the scattered parts along
`concat_dimension` and produces `results` tensors.
The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)` if `channel_id <= 0`.
* `cross_partition(replica_groups)` if `channel_id > 0`.

Afterwards, within each `process_group`:

* `split_parts...@sender = split(operands...@sender, split_count, split_dimension)`
  for all `sender` in `process_group`.
* `scattered_parts...@receiver = [split_parts...@sender[receiver_index] for
  sender in process_group]` where
  `receiver_index = process_group.index(receiver)`.
* `results...@process = concatenate(scattered_parts...@process, concat_dimension)`.

#### Inputs

| Label | Name               | Type                                                         | Constraints            |
|-------|--------------------|--------------------------------------------------------------|------------------------|
| (I1)  | `operands`         |  variadic number of tensors or per-tensor quantized tensors  | (C1-C3), (C9)          |
| (I2)  | `split_dimension`  | constant of type `si64`                                      | (C1), (C2), (C9)       |
| (I3)  | `concat_dimension` | constant of type `si64`                                      | (C3), (C9)             |
| (I4)  | `split_count`      | constant of type `si64`                                      | (C2), (C4), (C8), (C9) |
| (I5)  | `replica_groups`   | 2-dimensional tensor constant of type `si64`                 | (C5-C8)                |
| (I6)  | `channel_id`       | constant of type `si64`                                      |                        |

#### Outputs

| Name      | Type                                                        | Constraints |
|-----------|-------------------------------------------------------------|-------------|
| `results` | variadic number of tensors or per-tensor quantized tensors  | (C9)        |

#### Constraints

* (C1) `0 <= split_dimension < rank(operands...)`.
* (C2) `dim(operands..., split_dimension) % split_count = 0`.
* (C3) `0 <= concat_dimension < rank(operands...)`.
* (C4) `0 < split_count`.
* (C5) `is_unique(replica_groups)`.
* (C6) `size(replica_groups)` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_partitions` if `cross_partition` is used.
* (C7) `0 <= replica_groups < size(replica_groups)`.
* (C8) `dim(replica_groups, 1) = split_count`.
* (C9) `type(results...) = type(operands...)` except, if `split_dimension !=
  concat_dimension`:
  * `dim(results..., split_dimension) =
    dim(operands..., split_dimension) / split_count`.
  * `dim(results..., concat_dimension) =
    dim(operands..., concat_dimension) * split_count`.

#### Examples

```mlir
// num_replicas: 2
// num_partitions: 1
// %operand1@(0, 0): [[1, 2, 3, 4],
//                    [5, 6, 7, 8]]
// %operand1@(1, 0): [[9, 10, 11, 12],
//                    [13, 14, 15, 16]]
// %operand2@(0, 0): [[17, 18, 19, 20],
//                    [21, 22, 23, 24]]
// %operand2@(1, 0): [[25, 26, 27, 28],
//                    [29, 30, 31, 32]]
%result:2 = "stablehlo.all_to_all"(%operand1, %operand2) {
  split_dimension = 1 : i64,
  concat_dimension = 0 : i64,
  split_count = 2 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // channel_id = 0
} : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
// %result#0@(0, 0): [[1, 2], [5, 6], [9, 10], [13, 14]]
// %result#0@(1, 0): [[3, 4], [7, 8], [11, 12], [15, 16]]
// %result#1@(0, 0): [[17, 18], [21, 22], [25, 26], [29, 30]]
// %result#1@(1, 0): [[19, 20], [23, 24], [27, 28], [31, 32]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/all_to_all.mlir)

### and

#### Semantics

Performs element-wise AND of two tensors `lhs` and `rhs` and produces a `result`
tensor. Depending on the element type, does the following:

* For booleans: logical AND.
* For integers: bitwise AND.

#### Inputs

| Label | Name  | Type                              | Constraints |
|-------|-------|-----------------------------------|-------------|
| (I1)  | `lhs` | tensor of boolean or integer type | (C1)        |
| (I2)  | `rhs` | tensor of boolean or integer type | (C1)        |

#### Outputs

| Name     | Type                              | Constraints |
|----------|-----------------------------------|-------------|
| `result` | tensor of boolean or integer type | (C1)        |

#### Constraints

* (C1) `type(lhs) = type(rhs) = type(result)`.

#### Examples

```mlir
// %lhs: [[1, 2], [3, 4]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.and"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[1, 2], [3, 0]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/and.mlir)

### atan2

#### Semantics

Performs element-wise atan2 operation on `lhs` and `rhs` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `atan2` from IEEE-754.
* For complex numbers: complex atan2.
* For quantized types: `dequantize_op_quantize(atan2, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                                                                    | Constraints |
|-------|-------|-------------------------------------------------------------------------|-------------|
| (I1)  | `lhs` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(lhs) = baseline_type(rhs) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [0.0, 1.0, -1.0]
// %rhs: [0.0, 0.0, 0.0]
%result = "stablehlo.atan2"(%lhs, %rhs) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
// %result: [0.0, 1.57079637, -1.57079637] // [0.0, pi/2, -pi/2]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/atan2.mlir)

### batch_norm_grad

#### Semantics

Computes gradients of several inputs of `batch_norm_training` backpropagating
from `grad_output`, and produces `grad_operand`, `grad_scale` and `grad_offset`
tensors. More formally, this operation can be expressed as a decomposition to
existing StableHLO operations using Python syntax as follows:

```python
def compute_sum(operand, feature_index):
  (sum,) = reduce(
      inputs=[operand],
      init_values=[constant(0, element_type(operand))],
      dimensions=[i for i in range(rank(operand)) if i != feature_index],
      body=lambda x, y: add(x, y))
  return sum

def compute_mean(operand, feature_index):
  sum = compute_sum(operand, feature_index)
  divisor = constant(size(operand) / dim(operand, feature_index),
                     element_type(operand))
  divisor_bcast = broadcast_in_dim(divisor, [], type(sum))
  return divide(sum, divisor_bcast)

def batch_norm_grad(operand, scale, mean, variance, grad_output, epsilon, feature_index):
  # Broadcast inputs to type(operand)
  scale_bcast = broadcast_in_dim(scale, [feature_index], type(operand))
  mean_bcast = broadcast_in_dim(mean, [feature_index], type(operand))
  variance_bcast = broadcast_in_dim(variance, [feature_index], type(operand))
  epsilon_bcast = broadcast_in_dim(constant(epsilon, element_type(operand)), [],
                                   type(operand))

  # Perform normalization using the provided `mean` and `variance`
  # Intermediate values will be useful for computing gradients
  centered_operand = subtract(operand, mean_bcast)
  stddev = sqrt(add(variance_bcast, epsilon_bcast))
  normalized_operand = divide(centered_operand, stddev)

  # Use the implementation from batchnorm_expander.cc in XLA
  # Temporary variables have exactly the same names as in the C++ code
  elements_per_feature = broadcast_in_dim(
      constant(divide(size(operand), dim(operand, feature_index)),
               element_type(grad_output)),
      [], type(operand))
  i1 = multiply(grad_output, elements_per_feature)
  i2 = broadcast_in_dim(
      compute_sum(grad_output, feature_index), [feature_index], type(operand))
  i3 = broadcast_in_dim(
      compute_sum(multiply(grad_output, centered_operand), feature_index),
      [feature_index], type(operand))
  i4 = multiply(i3, centered_operand)
  i5 = divide(i4, add(variance_bcast, epsilon_bcast))
  i6 = subtract(subtract(i1, i2), i5)

  grad_operand =
      multiply(divide(divide(scale_bcast, stddev), elements_per_feature), i6)
  grad_scale =
      compute_sum(multiply(grad_output, normalized_operand), feature_index)
  grad_offset = compute_sum(grad_output, feature_index)

  return grad_operand, grad_scale, grad_offset
```

For quantized types, performs
`dequantize_batch_norm_grad_or_training_quantize(lambda operand, scale, mean,
variance, grad_output: batch_norm_grad(operand, scale, mean, variance,
grad_output, epsilon, feature_index), operand, scale, mean, variance,
grad_output, type(grad_operand), type(grad_scale), type(feature_index))`.

#### Inputs

| Label | Name            | Type                                                                | Constraints      |
|-------|-----------------|---------------------------------------------------------------------|------------------|
| (I1)  | `operand`       | tensor of floating-point type or per-tensor quantized tensor        | (C1-C3), (C5)    |
| (I2)  | `scale`         | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C4), (C5) |
| (I3)  | `mean`          | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C4)       |
| (I4)  | `variance`      | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C4)       |
| (I5)  | `grad_output`   | tensor of floating-point type or per-tensor quantized tensor        | (C2), (C3)       |
| (I6)  | `epsilon`       | constant of type `f32`                                              |                  |
| (I7)  | `feature_index` | constant of type `si64`                                             | (C1), (C5)       |

#### Outputs

| Name           | Type                                                                | Constraints |
|----------------|---------------------------------------------------------------------|-------------|
| `grad_operand` | tensor of floating-point type or per-tensor quantized tensor        | (C2), (C3)  |
| `grad_scale`   | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C4)  |
| `grad_offset`  | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C4)  |

#### Constraints

* (C1) `0 <= feature_index < rank(operand)`.
* (C2) `operand`, `scale`, `mean`, `variance`, `grad_output`, `grad_operand`,
       `grad_scale` and `grad_offset` have the same `baseline_element_type`.
* (C3) `operand`, `grad_output` and `grad_operand` have the same shape.
* (C4) `scale`, `mean`, `variance`, `grad_scale` and `grad_offset` have the
       same shape.
* (C5) `size(scale) = dim(operand, feature_index)`.

#### Examples

```mlir
// %operand: [
//            [[1.0, 2.0], [3.0, 4.0]],
//            [[3.0, 4.0], [1.0, 2.0]]
//           ]
// %scale: [1.0, 1.0]
// %mean: [2.0, 3.0]
// %variance: [1.0, 1.0]
// %grad_output: [
//                [[0.1, 0.1], [0.1, 0.1]],
//                [[0.1, 0.1], [0.1, 0.1]]
//               ]
%grad_operand, %grad_scale, %grad_offset =
"stablehlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
  epsilon = 0.0 : f32,
  feature_index = 2 : i64
} : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>,
     tensor<2x2x2xf64>) -> (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
// %grad_operand: [
//                 [[0.0, 0.0], [0.0, 0.0]],
//                 [[0.0, 0.0], [0.0, 0.0]]
//                ]
// %grad_scale:  [0.0, 0.0]
// %grad_offset: [0.4, 0.4]
```

### batch_norm_inference

#### Semantics

Normalizes the `operand` tensor across all dimensions except for the
`feature_index` dimension and produces a `result` tensor. More formally, this
operation can be expressed as a decomposition to existing StableHLO operations
using Python syntax as follows:

```python
def batch_norm_inference(operand, scale, offset, mean, variance, epsilon, feature_index):
  # Broadcast inputs to shape(operand)
  scale_bcast = broadcast_in_dim(scale, [feature_index], type(operand))
  offset_bcast = broadcast_in_dim(offset, [feature_index], type(operand))
  mean_bcast = broadcast_in_dim(mean, [feature_index], type(operand))
  variance_bcast = broadcast_in_dim(variance, [feature_index], type(operand))
  epsilon_bcast = broadcast_in_dim(constant(epsilon, element_type(operand)), [],
                                   type(operand))

  # Perform normalization using the provided `mean` and `variance` instead of
  # computing them like `batch_norm_training` does.
  centered_operand = subtract(operand, mean_bcast)
  stddev = sqrt(add(variance_bcast, epsilon_bcast))
  normalized_operand = divide(centered_operand, stddev)
  return add(multiply(scale_bcast, normalized_operand), offset_bcast)
```

For quantized types, performs
`dequantize_op_quantize(lambda operand, scale, offset, mean, variance:
batch_norm_inference(operand, scale, offset, mean, variance, epsilon,
feature_index), operand, scale, offset, mean, variance, type(result))`.

#### Inputs

| Label | Name            | Type                                                                | Constraints   |
|-------|-----------------|---------------------------------------------------------------------|---------------|
| (I1)  | `operand`       | tensor of floating-point type or per-tensor quantized tensor        | (C1-C7)       |
| (I2)  | `scale`         | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C3)    |
| (I3)  | `offset`        | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C4)    |
| (I4)  | `mean`          | 1-dimensional tensor of floating-point or per-tensor quantized type | (C5)          |
| (I5)  | `variance`      | 1-dimensional tensor of floating-point or per-tensor quantized type | (C2), (C6)    |
| (I6)  | `epsilon`       | constant of type `f32`                                              |               |
| (I7)  | `feature_index` | constant of type `si64`                                             | (C1), (C3-C6) |

#### Outputs

| Name     | Type                                                         | Constraints |
|----------|--------------------------------------------------------------|-------------|
| `result` | tensor of floating-point type or per-tensor quantized tensor | (C2), (C7)  |

#### Constraints

* (C1) `0 <= feature_index < rank(operand)`.
* (C2) `operand`, `scale`, `offset`, `mean`, `variance` and `result` have the
       same `baseline_element_type`.
* (C3) `size(scale) = dim(operand, feature_index)`.
* (C4) `size(offset) = dim(operand, feature_index)`.
* (C5) `size(mean) = dim(operand, feature_index)`.
* (C6) `size(variance) = dim(operand, feature_index)`.
* (C7) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [
//            [[1.0, 2.0], [3.0, 4.0]],
//            [[3.0, 4.0], [1.0, 2.0]]
//           ]
// %scale: [1.0, 1.0]
// %offset: [1.0, 1.0]
// %mean: [2.0, 3.0]
// %variance: [1.0, 1.0]
%result = "stablehlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) {
  epsilon = 0.0 : f32,
  feature_index = 2 : i64
} : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>) -> tensor<2x2x2xf64>
// %result: [
//           [[0.0, 0.0], [2.0, 2.0]],
//           [[2.0, 2.0], [0.0, 0.0]]
//          ]
```

### batch_norm_training

#### Semantics

Computes mean and variance across all dimensions except for the `feature_index`
dimension and normalizes the `operand` tensor producing `output`, `batch_mean`
and `batch_var` tensors. More formally, this operation can be expressed as a
decomposition to existing StableHLO operations using Python syntax as
follows:

```python
def compute_mean(operand, feature_index):
  (sum,) = reduce(
      inputs=[operand],
      init_values=[constant(0, element_type(operand))],
      dimensions=[i for i in range(rank(operand)) if i != feature_index],
      body=lambda x, y: add(x, y))
  divisor = constant(size(operand) / dim(operand, feature_index),
                     element_type(operand))
  divisor_bcast = broadcast_in_dim(divisor, [], type(sum))
  return divide(sum, divisor_bcast)

def compute_variance(operand, feature_index):
  mean = compute_mean(operand, feature_index)
  mean_bcast = broadcast_in_dim(mean, [feature_index], type(operand))
  centered_operand = subtract(operand, mean_bcast)
  return compute_mean(mul(centered_operand, centered_operand), feature_index)

def batch_norm_training(operand, scale, offset, epsilon, feature_index):
  mean = compute_mean(operand, feature_index)
  variance = compute_variance(operand, feature_index)
  return batch_norm_inference(operand, scale, offset, mean, variance, epsilon,
                              feature_index),
         mean, variance
```

For quantized types, performs
`dequantize_batch_norm_grad_or_training_quantize(lambda operand, scale, offset:
batch_norm_training(operand, scale, offset, epsilon, feature_index), operand,
scale, offset, type(output), type(batch_mean), type(batch_var))`.

#### Inputs

| Label | Name            | Type                                                           | Constraints   |
|-------|-----------------|----------------------------------------------------------------|---------------|
| (I1)  | `operand`       | tensor of floating-point type or per-tensor quantized tensor   | (C1)          |
| (I2)  | `scale`         | 1-dimensional tensor of floating-point or per-tensor quantized | (C2), (C3)    |
| (I3)  | `offset`        | 1-dimensional tensor of floating-point or per-tensor quantized | (C2), (C4)    |
| (I4)  | `epsilon`       | constant of type `f32`                                         | (C1), (C3-C6) |
| (I5)  | `feature_index` | constant of type `si64`                                        | (C1), (C3-C6) |

#### Outputs

| Name         | Type                                                           | Constraints |
|--------------|----------------------------------------------------------------|-------------|
| `output`     | tensor of floating-point type or per-tensor quantized tensor   | (C7)        |
| `batch_mean` | 1-dimensional tensor of floating-point or per-tensor quantized | (C2), (C5)  |
| `batch_var`  | 1-dimensional tensor of floating-point or per-tensor quantized | (C2), (C6)  |

#### Constraints

* (C1) `0 <= feature_index < rank(operand)`.
* (C2) `operand`, `scale`, `offset`, `batch_mean`, `batch_var` and `output` have
       the same `baseline_element_type`.
* (C3) `size(scale) = dim(operand, feature_index)`.
* (C4) `size(offset) = dim(operand, feature_index)`.
* (C5) `size(batch_mean) = dim(operand, feature_index)`.
* (C6) `size(batch_var) = dim(operand, feature_index)`.
* (C7) `baseline_type(output) = baseline_type(operand)`.

#### Examples

```mlir
// %operand: [
//            [[1.0, 2.0], [3.0, 4.0]],
//            [[3.0, 4.0], [1.0, 2.0]]
//           ]
// %scale: [1.0, 1.0]
// %offset: [1.0, 1.0]
%output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%operand, %scale, %offset) {
  epsilon = 0.0 : f32,
  feature_index = 2 : i64
} : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>) ->
    (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
// %output: [
//           [[0.0, 0.0], [2.0, 2.0]],
//           [[2.0, 2.0], [0.0, 0.0]]
//          ]
// %batch_mean: [2.0, 3.0]
// %batch_var: [1.0, 1.0]
```

### bitcast_convert

#### Semantics

Performs a bitcast operation on `operand` tensor and produces a `result` tensor
where the bits of the entire `operand` tensor are reinterpreted using the
type of the `result` tensor.

More formally, given `E = element_type(operand)`, `E' = element_type(result)`,
and `R = rank(operand)`:

* If `num_bits(E') < num_bits(E)`,
  `bits(result[i0, ..., iR-1, :]) = bits(operand[i0, ..., iR-1])`.
* If `num_bits(E') > num_bits(E)`,
  `bits(result[i0, ..., iR-2]) = bits(operand[i0, ..., iR-2, :])`.
* If `num_bits(E') = num_bits(E)`,
  `bits(result[i0, ..., iR-1]) = bits(operand[i0, ..., iR-1])`.

`bits` returns in-memory representation of a given value, and its behavior
is implementation-defined because the exact representation of tensors is
implementation-defined, and the exact representation of element types is
implementation-defined as well.

#### Inputs

| Label | Name      | Type                       | Constraints |
|-------|-----------|----------------------------|-------------|
| (I1)  | `operand` | tensor or quantized tensor | (C1-C2)     |

#### Outputs

| Name     | Type                       | Constraints |
|----------|----------------------------|-------------|
| `result` | tensor or quantized tensor | (C1-C2)     |

#### Constraints

* (C1) Given `E = is_quantized(operand) ? storage_type(operand) :
  element_type(operand)`, `E' = is_quantized(result) ?
  storage_type(result) : element_type(result)`, and `R = rank(operand)`:
  * If `num_bits(E') = num_bits(E)`, `shape(result) = shape(operand)`.
  * If `num_bits(E') < num_bits(E)`:
    * `rank(result) = R + 1`.
    * `dim(result, i) = dim(operand, i)` for all `0 <= i < R`.
    * `dim(result, R) * num_bits(E') = num_bits(E)`.
  * If `num_bits(E') > num_bits(E)`:
    * `rank(result) = R - 1`.
    * `dim(result, i) = dim(operand, i)` for all `0 <= i < R`.
    * `dim(operand, R - 1) * num_bits(E) = num_bits(E')`.
* (C2) If `is_complex(operand) or is_complex(result)`, then
  `is_complex(operand) and is_complex(result)`.

#### Examples

```mlir
// %operand: 0x0123456789ABCDEF
%result = "stablehlo.bitcast_convert"(%operand) : (tensor<f64>) -> tensor<4xf16>
// %result: [0xCDEF, 0x89AB, 0x4567, 0x0123] // little-endian representation
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/bitcast_convert.mlir)

### broadcast_in_dim

#### Semantics

Expands the dimensions and/or rank of an input tensor by duplicating the data
in the `operand` tensor and produces a `result` tensor. More formally,
`result[result_index] = operand[operand_index]` where for all `d` in
`axes(operand)`:

* `operand_index[d] = 0` if `dim(operand, d) = 1`.
* `operand_index[d] = result_index[broadcast_dimensions[d]]` otherwise.

#### Inputs

| Label | Name                   | Type                                         | Constraints      |
|-------|------------------------|----------------------------------------------|------------------|
| (I1)  | `operand`              | tensor or quantized tensor                   | (C1-C2), (C5-C6) |
| (I2)  | `broadcast_dimensions` | 1-dimensional tensor constant of type `si64` | (C2-C6)          |

#### Outputs

| Name     | Type                       | Constraints         |
|----------|----------------------------|---------------------|
| `result` | tensor or quantized tensor | (C1), (C3), (C5-C6) |

#### Constraints

* (C1) `element_type(result)` is given by:
  * `element_type(operand)`, if `!is_per_axis_quantized(operand)`.
  * `element_type(operand)` except that `quantization_dimension(operand)`,
  `scales(operand)`, and `zero_points(operand)` may differ from
  `quantization_dimension(result)`, `scales(result)`, and `zero_points(result)`
  resp., otherwise.
* (C2) `size(broadcast_dimensions) = rank(operand)`.
* (C3) `0 <= broadcast_dimensions < rank(result)`.
* (C4) `is_unique(broadcast_dimensions)`.
* (C5) For all `d` in `axes(operand)`:
  * `dim(operand, d) = 1` or
  * `dim(operand, d) = dim(result, broadcast_dimensions[d])`.
* (C6) If `is_per_axis_quantized(result)`:
  * `quantization_dimension(result) = broadcast_dimensions[quantization_dimension(operand)]`.
  * If `dim(operand, quantization_dimension(operand)) = 1`, then
    `scales(result)[i] = scales(operand)[0] and zero_points(result)[i] =
    zero_points(operand)[0] for i in
    range(dim(result, quantization_dimension(result)))`.

#### Examples

```mlir
// %operand: [
//            [1, 2, 3]
//           ]
%result = "stablehlo.broadcast_in_dim"(%operand) {
  broadcast_dimensions = array<i64: 2, 1>
} : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
// %result: [
//            [
//             [1, 1],
//             [2, 2],
//             [3, 3]
//            ],
//            [
//             [1, 1],
//             [2, 2],
//             [3, 3]
//            ]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/broadcast_in_dim.mlir)

### case

#### Semantics

Produces the output from executing exactly one function from `branches`
depending on the value of `index`. More formally, `result = selected_branch()`
where:

* `selected_branch = branches[index]` if `0 <= index < size(branches)`.
* `selected_branch = branches[-1]` otherwise.

#### Inputs

| Label | Name       | Type                                | Constraints |
|-------|------------|-------------------------------------|-------------|
| (I1)  | `index`    | 0-dimensional tensor of type `si32` |             |
| (I2)  | `branches` | variadic number of functions        | (C1-C4)     |

#### Outputs

| Name      | Type                                                    | Constraints |
|-----------|---------------------------------------------------------|-------------|
| `results` | variadic number of tensors, quantized tensors or tokens | (C4)        |

#### Constraints

* (C1) `0 < size(branches)`.
* (C2) `input_types(branches...) = []`.
* (C3) `same(output_types(branches...))`.
* (C4) `type(results...) = output_types(branches[0])`.

#### Examples

```mlir
// %index: -1
// %result_branch0: [0, 0]
// %result_branch1: [1, 1]
%result0, %result1 = "stablehlo.case"(%index) ({
  "stablehlo.return"(%result_branch0, %result_branch0) : (tensor<2xi64>, tensor<2xi64>) -> ()
}, {
  "stablehlo.return"(%result_branch1, %result_branch1) : (tensor<2xi64>, tensor<2xi64>) -> ()
}) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
// %result0: [1, 1]
// %result1: [1, 1]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/case.mlir)

### cbrt

#### Semantics

Performs element-wise cubic root operation on `operand` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `rootn(x, 3)` from IEEE-754.
* For complex numbers: complex cubic root.
* For quantized types: `dequantize_op_quantize(cbrt, operand, type(result))`

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [0.0, 1.0, 8.0, 27.0]
%result = "stablehlo.cbrt"(%operand) : (tensor<4xf64>) -> tensor<4xf64>
// %result: [0.0, 1.0, 2.0, 3.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/cbrt.mlir)

### ceil

#### Semantics

Performs element-wise ceil of `operand` tensor and produces a `result` tensor.
Implements the `roundToIntegralTowardPositive` operation from the IEEE-754
specification. For quantized types, performs
`dequantize_op_quantize(ceil, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                         | Constraints |
|-------|-----------|--------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                         | Constraints |
|----------|--------------------------------------------------------------|-------------|
| `result` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [-0.8166, -0.2530, 0.2530, 0.8166, 2.0]
%result = "stablehlo.ceil"(%operand) : (tensor<5xf32>) -> tensor<5xf32>
// %result: [-0.0, -0.0, 1.0, 1.0, 2.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/ceil.mlir)

### cholesky

#### Semantics

Computes the Cholesky decomposition of a batch of matrices.

More formally, for all `i` in `index_space(result)`,
`result[i0, ..., iR-3, :, :]` is a Cholesky decomposition of
`a[i0, ..., iR-3, :, :]`, in the form of either of a lower-triangular
(if `lower` is `true`) or upper-triangular (if `lower` is `false`) matrix.
The output values in the opposite triangle, i.e. the strict upper triangle or
strict lower triangle correspondingly, are implementation-defined.

If there exists `i` where the input matrix is not an Hermitian positive-definite
matrix, then the behavior is undefined.

For quantized types, performs
`dequantize_op_quantize(lambda operand: cholesky(operand, lower), a, type(result))`.

#### Inputs

| Label | Name    | Type                                                                    | Constraints |
|-------|---------|-------------------------------------------------------------------------|-------------|
| (I1)  | `a`     | tensor of floating-point or complex type or per-tensor quantized tensor | (C1-C3)     |
| (I2)  | `lower` | 0-dimensional tensor constant of type `i1`                              |             |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(a) = baseline_type(result)`.
* (C2) `2 <= rank(a)`.
* (C3) `dim(a, -2) = dim(a, -1)`.

#### Examples

```mlir
// %a: [
//      [1.0, 2.0, 3.0],
//      [2.0, 20.0, 26.0],
//      [3.0, 26.0, 70.0]
//     ]
%result = "stablehlo.cholesky"(%a) {
  lower = true
} : (tensor<3x3xf32>) -> tensor<3x3xf64>
// %result: [
//           [1.0, 0.0, 0.0],
//           [2.0, 4.0, 0.0],
//           [3.0, 5.0, 6.0]
//          ]
```

### clamp

#### Semantics

Clamps every element of the `operand` tensor between a minimum and maximum
value and produces a `result` tensor. More formally, `result[result_index] =
minimum(maximum(operand[result_index], min_element), max_element)`,
where `min_element = rank(min) = 0 ? min[] : min[result_index]`,
`max_element = rank(max) = 0 ? max[] : max[result_index]`. For quantized types,
performs `dequantize_op_quantize(clamp, min, operand, max, type(result))`.

Imposing an ordering on complex numbers involves surprising semantics,
so in the future we are planning to remove support for complex numbers
for this operation ([#560](https://github.com/openxla/stablehlo/issues/560)).

#### Inputs

| Label | Name      | Type                                  | Constraints |
|-------|-----------|---------------------------------------|-------------|
| (I1)  | `min`     | tensor or per-tensor quantized tensor | (C1), (C3)  |
| (I2)  | `operand` | tensor or per-tensor quantized tensor | (C1-C4)     |
| (I3)  | `max`     | tensor or per-tensor quantized tensor | (C2), (C3)  |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C4)        |

#### Constraints

* (C1) `rank(min) = 0 or shape(min) = shape(operand)`.
* (C2) `rank(max) = 0 or shape(max) = shape(operand)`.
* (C3) `baseline_element_type(min) = baseline_element_type(operand) = baseline_element_type(max)`.
* (C4) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %min: [5, 10, 15]
// %operand: [3, 13, 23]
// %max: [10, 15, 20]
%result = "stablehlo.clamp"(%min, %operand, %max) : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
// %result: [5, 13, 20]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/clamp.mlir)

### collective_broadcast

#### Semantics

Within each process group in the StableHLO process grid, send the value of the
`operand` tensor from the source process to the target processes and produce a
`result` tensor.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)` if `channel_id <= 0`.
* `cross_partition(replica_groups)` if `channel_id > 0`.

Afterwards, `result@process` is given by:

* `operand@process_groups[i, 0]` if there exists an `i` such that the process is
  in `process_groups[i]`.
* `broadcast_in_dim(constant(is_quantized(result) ? quantize(0,
  element_type(result)) : 0, element_type(result)), [], type(result))`
  otherwise.

#### Inputs

| Label | Name             | Type                                                             | Constraints |
|-------|------------------|------------------------------------------------------------------|-------------|
| (I1)  | `operand`        | tensor or per-tensor quantized tensor                            | (C3)        |
| (I2)  | `replica_groups` | variadic number of 1-dimensional tensor constants of type `si64` | (C1), (C2)  |
| (I3)  | `channel_id`     | constant of type `si64`                                          |             |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C3)        |

#### Constraints

* (C1) `is_unique(replica_groups)`.
* (C2) `0 <= replica_groups < N` where `N` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_partitions` if `cross_partition` is used.
* (C3) `type(result) = type(operand)`.

#### Examples

```mlir
// num_replicas: 4
// num_partitions: 1
// %operand@(0, 0): [[1, 2]]
// %operand@(1, 0): [[3, 4]]
// %operand@(2, 0): [[5, 6]]
// %operand@(3, 0): [[7, 8]]
%result = "stablehlo.collective_broadcast"(%operand) {
  replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor1x2xi64>) -> tensor<1x2xi64>
// %result@(0, 0): [[0, 0]]
// %result@(1, 0): [[5, 6]]
// %result@(2, 0): [[5, 6]]
// %result@(3, 0): [[0, 0]]
```

### collective_permute

#### Semantics

Within each process group in the StableHLO process grid, sends the value of the
`operand` tensor from the source process to the target process and produces a
`result` tensor.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(source_target_pairs)` if `channel_id <= 0`.
* `cross_partition(source_target_pairs)` if `channel_id > 0`.

Afterwards, `result@process` is given by:

* `operand@process_groups[i, 0]`, if there exists an `i` such that
  `process_groups[i, 1] = process`.
* `broadcast_in_dim(constant(is_quantized(result) ? quantize(0,
  element_type(result)) : 0, element_type(result)), [], type(result))`
  otherwise.

#### Inputs

| Label | Name                  | Type                                         | Constraints |
|-------|-----------------------|----------------------------------------------|-------------|
| (I1)  | `operand`             | tensor or per-tensor quantized tensor        | (C5)        |
| (I2)  | `source_target_pairs` | 2-dimensional tensor constant of type `si64` | (C1-C4)     |
| (I3)  | `channel_id`          | constant of type `si64`                      |             |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `dim(source_target_pairs, 1) = 2`.
* (C2) `is_unique(source_target_pairs[:, 0])`.
* (C3) `is_unique(source_target_pairs[:, 1])`.
* (C4) `0 <= source_target_pairs < N`, where `N` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_partitions` if `cross_partition` is used.
* (C5) `type(result) = type(operand)`.

#### Examples

```mlir
// num_replicas: 3
// num_partitions: 1
// %operand@(0, 0): [[1, 2], [3, 4]]
// %operand@(1, 0): [[5, 6], [7, 8]]
// %operand@(2, 0): [[9, 10], [11, 12]]
%result = "stablehlo.collective_permute"(%operand) {
  source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x2xi64>) -> tensor<2x2xi64>
//
// %result@(0, 0): [[0, 0], [0, 0]]
// %result@(1, 0): [[1, 2], [3, 4]]
// %result@(2, 0): [[5, 6], [7, 8]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/collective_permute.mlir)

### compare

#### Semantics

Performs element-wise comparison of `lhs` and `rhs` tensors according to
`comparison_direction` and `compare_type`, and produces a `result` tensor.

The values of `comparison_direction` and `compare_type` have the following
semantics:

For boolean and integer element types:

* `EQ`: `lhs = rhs`.
* `NE`: `lhs != rhs`.
* `GE`: `lhs >= rhs`.
* `GT`: `lhs > rhs`.
* `LE`: `lhs <= rhs`.
* `LT`: `lhs < rhs`.

For floating-point element types with `compare_type = FLOAT`, the op implements
the following IEEE-754 operations:

* `EQ`: `compareQuietEqual`.
* `NE`: `compareQuietNotEqual`.
* `GE`: `compareQuietGreaterEqual`.
* `GT`: `compareQuietGreater`.
* `LE`: `compareQuietLessEqual`.
* `LT`: `compareQuietLess`.

For floating-point element types with `compare_type = TOTALORDER`, the op
uses the combination of `totalOrder` and `compareQuietEqual` operations from
IEEE-754.

For complex element types, lexicographic comparison of `(real, imag)` pairs is
performed using the provided `comparison_direction` and `compare_type`.
Imposing an ordering on complex numbers involves surprising semantics,
so in the future we are planning to remove support for complex numbers
when `comparison_direction` is `GE`, `GT`, `LE` or `LT`
([#560](https://github.com/openxla/stablehlo/issues/560)).

For quantized types. performs `dequantize_compare(lhs, rhs,
comparison_direction)`.

#### Inputs

| Label | Name                   | Type                                                    | Constraints |
|-------|------------------------|---------------------------------------------------------|-------------|
| (I1)  | `lhs`                  | tensor or per-tensor quantized tensor                   | (C1-C3)     |
| (I2)  | `rhs`                  | tensor or per-tensor quantized tensor                   | (C1-C2)     |
| (I3)  | `comparison_direction` | enum of `EQ`, `NE`, `GE`, `GT`, `LE`, and `LT`          |             |
| (I4)  | `compare_type`         | enum of `FLOAT`, `TOTALORDER`, `SIGNED`, and `UNSIGNED` | (C3)        |

#### Outputs

| Name     | Type                   | Constraints |
|----------|------------------------|-------------|
| `result` | tensor of boolean type | (C2)        |

#### Constraints

* (C1) `baseline_element_type(lhs) = baseline_element_type(rhs)`.
* (C2) `shape(lhs) = shape(rhs) = shape(result)`.
* (C3) `compare_type` is defined as:
  * `SIGNED` if `is_signed_integer(element_type(lhs))`.
  * `UNSIGNED` if `is_unsigned_integer(element_type(lhs)) or
    is_boolean(element_type(lhs))`.
  * `FLOAT` or `TOTALORDER` if `is_float(element_type(lhs))`.
  * `FLOAT` if `is_complex(element_type(lhs))`.

#### Examples

```mlir
// %lhs: [1.0, 3.0]
// %rhs: [1.1, 2.9]
%result = "stablehlo.compare"(%lhs, %rhs) {
  comparison_direction = #stablehlo<comparison_direction LT>,
  compare_type = #stablehlo<comparison_type FLOAT>
} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// %result: [true, false]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/compare.mlir)

### complex

#### Semantics

Performs element-wise conversion to a complex value from a pair of real and
imaginary values, `lhs` and `rhs`, and produces a `result` tensor.

#### Inputs

| Label | Name  | Type                          | Constraints |
|-------|-------|-------------------------------|-------------|
| (I1)  | `lhs` | tensor of type `f32` or `f64` | (C1-C3)     |
| (I2)  | `rhs` | tensor of type `f32` or `f64` | (C1)        |

#### Outputs

| Name     | Type                   | Constraints |
|----------|------------------------|-------------|
| `result` | tensor of complex type | (C2), (C3)  |

#### Constraints

* (C1) `type(lhs) = type(rhs)`.
* (C2) `shape(result) = shape(lhs)`.
* (C3) `element_type(result)` has type `complex<E>` where
  `E = element_type(lhs)`.

#### Examples

```mlir
// %lhs: [1.0, 3.0]
// %rhs: [2.0, 4.0]
%result = "stablehlo.complex"(%lhs, %rhs) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xcomplex<f64>>
// %result: [(1.0, 2.0), (3.0, 4.0)]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/complex.mlir)

### composite

#### Semantics

Encapsulates an operation made up (composed) of other StableHLO operations,
taking `inputs` and `composite_attributes` and producing `results`. The
semantics of the op are implemented by the `decomposition` attribute. The
`composite` op can be replaced with its decomposition without changing program
semantics. In cases where inlining the decomposition does not provide the same
op semantics, prefer using `custom_call`.

The `version` field (defaults to `0`) is used to denote when a composite's
semantics change.

#### Inputs

| Label | Name                   | Type                      |
|-------|------------------------|---------------------------|
| (I1)  | `inputs`               | variadic number of values |
| (I2)  | `name`                 | constant of type `string` |
| (I3)  | `composite_attributes` | attribute dictionary      |
| (I4)  | `decomposition`        | constant of type `string` |
| (I5)  | `version`              | constant of type `si32`   |

#### Outputs

| Name      | Type                      |
|-----------|---------------------------|
| `results` | variadic number of values |

#### Constraints

* (C1) `is_namespaced_op_name(name)`
* (C2) `is_defined_in_parent_scope(decomposition)`
* (C3) `types(inputs...) == input_types(decomposition)`
* (C4) `types(results...) == output_types(decomposition)`

#### Examples

```mlir
%results = "stablehlo.composite"(%input0, %input1) {
  name = "my_namespace.my_op",
  composite_attributes = {
    my_attribute = "my_value"
  },
  decomposition = @my_op,
  version = 1 : i32
} : (tensor<f32>, tensor<f32>) -> tensor<f32>
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/composite.mlir)

### concatenate

#### Semantics

Concatenates `inputs` along `dimension` dimension in the same order as the given
arguments and produces a `result` tensor. More formally,
`result[i0, ..., id, ..., iR-1] = inputs[k][i0, ..., kd, ..., iR-1]`, where:

1. `id = d0 + ... + dk-1 + kd`.
1. `d` is equal to `dimension`, and `d0`, ... are `d`th dimension sizes
   of `inputs`.

#### Inputs

| Label | Name        | Type                                                       | Constraints      |
|-------|-------------|------------------------------------------------------------|------------------|
| (I1)  | `inputs`    | variadic number of tensors or per-tensor quantized tensors | (C1-C6)          |
| (I2)  | `dimension` | constant of type `si64`                                    | (C2), (C4), (C6) |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C5-C6)     |

#### Constraints

* (C1) `same(element_type(inputs...))`.
* (C2) `same(shape(inputs...))` except for `dim(inputs..., dimension)`.
* (C3) `0 < size(inputs)`.
* (C4) `0 <= dimension < rank(inputs[0])`.
* (C5) `element_type(result) = element_type(inputs[0])`.
* (C6) `shape(result) = shape(inputs[0])` except for:
  * `dim(result, dimension) = dim(inputs[0], dimension) + ...`.

#### Examples

```mlir
// %input0: [[1, 2], [3, 4], [5, 6]]
// %input1: [[7, 8]]
%result = "stablehlo.concatenate"(%input0, %input1) {
  dimension = 0 : i64
} : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
// %result: [[1, 2], [3, 4], [5, 6], [7, 8]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/concatenate.mlir)

### constant

#### Semantics

Produces an `output` tensor from a constant `value`.

#### Inputs

| Label | Name    | Type     | Constraints |
|-------|---------|----------|-------------|
| (I1)  | `value` | constant | (C1)        |

#### Outputs

| Name     | Type                       | Constraints |
|----------|----------------------------|-------------|
| `output` | tensor or quantized tensor | (C1)        |

#### Constraints

* (C1) `type(value) = type(output)`.

#### Examples

```mlir
%output = "stablehlo.constant"() {
  value = dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
} : () -> tensor<2x2xf32>
// %output: [[0.0, 1.0], [2.0, 3.0]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/constant.mlir)

### convert

#### Semantics

Performs an element-wise conversion from one element type to another on
`operand` tensor and produces a `result` tensor.

For **boolean-to-any-supported-type** conversions, the value `false` is
converted to zero, and the value `true` is converted to one. For
**any-supported-type-to-boolean** conversions, a zero value is converted to
`false`, and non-zero values are converted to `true`. See below for how this
work for complex types.

For conversions involving **integer-to-integer**, **integer-to-floating-point**
or **floating-point-to-floating-point**, if the source value can be exactly
represented in the destination type, the result value is that exact
representation. Otherwise, the behavior is TBD
([#180](https://github.com/openxla/stablehlo/issues/180)).

For conversions involving **floating-point-to-integer**, the fractional part is
truncated. If the truncated value cannot be represented in the destination type,
the behavior is TBD ([#180](https://github.com/openxla/stablehlo/issues/180)).

Conversion involving **complex-to-complex** follow the same behavior of
**floating-point-to-floating-point** conversions for converting real and
imaginary parts.

For **complex-to-any-other-type** and **any-other-type-to-complex** conversions,
the source imaginary value is ignored or the destination imaginary value is
zeroed, respectively. The conversion of the real part follows the
floating-point conversions.

In principle, this operation could express dequantization (conversion from
quantized tensors to regular tensors), quantization (conversion from regular
tensors to quantized tensors) and requantization (conversion between quantized
tensors), but at the moment we have dedicated operations for that -
`uniform_dequantize` for the first use case and `uniform_quantize` for the
second and the third use cases. In the future, these two ops may be merged
into `convert` ([#1576](https://github.com/openxla/stablehlo/issues/1576)).

#### Inputs

| Label | Name      | Type   | Constraints |
|-------|-----------|--------|-------------|
| (I1)  | `operand` | tensor | (C1)        |

#### Outputs

| Name     | Type   | Constraints |
|----------|--------|-------------|
| `result` | tensor | (C1)        |

#### Constraints

* (C1) `shape(operand) = shape(result)`.

#### Examples

```mlir
// %operand: [-1, 0, 1]
%result = "stablehlo.convert"(%operand) : (tensor<3xi64>) -> tensor<3xcomplex<f64>>
// %result: [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/convert.mlir)

### convolution

#### Semantics

Computes dot products between windows of `lhs` and slices of `rhs` and produces
`result`. The following diagram shows how elements in `result` are computed from
`lhs` and `rhs` using a concrete example.

![convolution](images/spec/convolution.svg)

More formally, consider the following reframing of the inputs in terms of `lhs`
in order to be able to express windows of `lhs`:

<!-- markdownlint-disable line-length -->
* `lhs_window_dimensions = lhs_shape(dim(lhs, input_batch_dimension), dim(rhs, kernel_spatial_dimensions), dim(lhs, input_feature_dimension))`.
* `lhs_window_strides = lhs_shape(1, window_strides, 1)`.
* `lhs_padding = lhs_shape([0, 0], padding, [0, 0])`.
* `lhs_base_dilations = lhs_shape(1, lhs_dilation, 1)`.
* `lhs_window_dilations = lhs_shape(1, rhs_dilation, 1)`.

This reframing uses the following helper functions:

* `lhs_shape(n, hw, c) = permute([n] + hw + [c], [input_batch_dimension] + input_spatial_dimensions + [input_feature_dimension])`.
* `result_shape(n1, hw, c1) = permute([n1] + hw + [c1], [output_batch_dimension] + output_spatial_dimensions + [output_feature_dimension])`.
* `permute([j0, j1, ..., jR-1], permutation) = [i0, i1, ..., iR-1]` where `j[d] = i[permutation[d]]`.

If `feature_group_count = 1` and `batch_group_count = 1`, then for all
`output_spatial_index` in `index_space(dim(result, output_spatial_dimensions...))`,
`result[result_shape(:, output_spatial_index, :)] = dot_product` where:

* `padding_value = constant(0, element_type(lhs))`.
* `padded_lhs = pad(lhs, padding_value, lhs_padding[:, 0], lhs_padding[:, 1], lhs_base_dilations - 1)`.
* `lhs_window_start = lhs_shape(0, output_spatial_index, 0) * lhs_window_strides`.
* `lhs_window = slice(padded_lhs, lhs_window_start, lhs_window_start + lhs_window_dimensions, lhs_window_dilations)`.
* `reversed_lhs_window = reverse(lhs_window, [input_spatial_dimensions[dim] for dim in range(size(window_reversal)) if window_reversal[dim] = true])`.
  This feature appears to be unused, so in the future we are planning to remove
  it ([#1181](https://github.com/openxla/stablehlo/issues/1181)).
* `dot_product = dot_general(reversed_lhs_window, rhs,
    lhs_batching_dimensions=[],
    lhs_contracting_dimensions=input_spatial_dimensions + [input_feature_dimension],
    rhs_batching_dimensions=[],
    rhs_contracting_dimensions=kernel_spatial_dimensions + [kernel_input_feature_dimension])`.

If `feature_group_count > 1`:

* `lhses = split(lhs, feature_group_count, input_feature_dimension)`.
* `rhses = split(rhs, feature_group_count, kernel_output_feature_dimension)`.
* `results... = convolution(lhses..., rhses..., ..., feature_group_count=1, ...)`.
* `result = concatenate(results, output_feature_dimension)`.

If `batch_group_count > 1`:

* `lhses = split(lhs, batch_group_count, input_batch_dimension)`.
* `rhses = split(rhs, batch_group_count, kernel_output_feature_dimension)`.
* `results... = convolution(lhses..., rhses..., ..., batch_group_count=1, ...)`.
* `result = concatenate(results, output_feature_dimension)`.
<!-- markdownlint-enable line-length -->

For quantized types, performs `dequantize_op_quantize(
    lambda lhs, rhs: convolution(lhs, rhs, window_strides, padding,
        lhs_dilation, rhs_dilation, window_reversal, input_batch_dimension,
        input_feature_dimension, input_spatial_dimensions,
        kernel_input_feature_dimension, kernel_output_feature_dimension,
        kernel_spatial_dimensions, output_batch_dimension,
        output_feature_dimension, output_spatial_dimensions,
        feature_group_count, batch_group_count, precision_config), lhs, rhs,
        type(result))`.

For hybrid quantized types, performs `hybrid_dequantize_then_op(
    lambda lhs, rhs: convolution(lhs, rhs, window_strides, padding,
        lhs_dilation, rhs_dilation, window_reversal, input_batch_dimension,
        input_feature_dimension, input_spatial_dimensions,
        kernel_input_feature_dimension, kernel_output_feature_dimension,
        kernel_spatial_dimensions, output_batch_dimension,
        output_feature_dimension, output_spatial_dimensions,
        feature_group_count, batch_group_count, precision_config), lhs, rhs)`.

#### Inputs

| Label | Name                              | Type                                                         | Constraints                                               |
|-------|-----------------------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| (I1)  | `lhs`                             | tensor or per-tensor quantized tensor                        | (C1), (C10-C11), (C14) (C25), (C27-C28), (C31-C32), (C34) |
| (I2)  | `rhs`                             | tensor or quantized tensor                                   | (C1), (C14-C16), (C25), (C27-C29), (C31-C34)              |
| (I3)  | `window_strides`                  | 1-dimensional tensor constant of type `si64`                 | (C2-C3), (C25)                                            |
| (I4)  | `padding`                         | 2-dimensional tensor constant of type `si64`                 | (C4), (C25)                                               |
| (I5)  | `lhs_dilation`                    | 1-dimensional tensor constant of type `si64`                 | (C5-C6), (C25)                                            |
| (I6)  | `rhs_dilation`                    | 1-dimensional tensor constant of type `si64`                 | (C7-C8), (C25)                                            |
| (I7)  | `window_reversal`                 | 1-dimensional tensor constant of type `i1`                   | (C9)                                                      |
| (I8)  | `input_batch_dimension`           | constant of type `si64`                                      | (C10), (C13), (C25)                                       |
| (I9)  | `input_feature_dimension`         | constant of type `si64`                                      | (C11), (C13-C14)                                          |
| (I10) | `input_spatial_dimensions`        | 1-dimensional tensor constant of type `si64`                 | (C12), (C13), (C25)                                       |
| (I11) | `kernel_input_feature_dimension`  | constant of type `si64`                                      | (C14), (C18)                                              |
| (I12) | `kernel_output_feature_dimension` | constant of type `si64`                                      | (C15-C16), (C18), (C25), (C29)                            |
| (I13) | `kernel_spatial_dimensions`       | 1-dimensional tensor constant of type `si64`                 | (C17-C18), (C25)                                          |
| (I14) | `output_batch_dimension`          | constant of type `si64`                                      | (C20), (C25)                                              |
| (I15) | `output_feature_dimension`        | constant of type `si64`                                      | (C20), (C25), (C30)                                       |
| (I16) | `output_spatial_dimensions`       | 1-dimensional tensor constant of type `si64`                 | (C19-C20), (C25)                                          |
| (I17) | `feature_group_count`             | constant of type `si64`                                      | (C11), (C14), (C16), (C21), (C23)                         |
| (I18) | `batch_group_count`               | constant of type `si64`                                      | (C10), (C15), (C22), (C23), (C25)                         |
| (I19) | `precision_config`                | variadic number of enums of `DEFAULT`, `HIGH`, and `HIGHEST` | (C24)                                                     |

#### Outputs

| Name     | Type                       | Constraints                |
|----------|----------------------------|----------------------------|
| `result` | tensor or quantized tensor | (C25-C28), (C30), (C32-34) |

#### Constraints

<!-- markdownlint-disable line-length -->
* (C1) `N = rank(lhs) = rank(rhs)`.
* (C2) `size(window_strides) = N - 2`.
* (C3) `0 < window_strides`.
* (C4) `shape(padding) = [N - 2, 2]`.
* (C5) `size(lhs_dilation) = N - 2`.
* (C6) `0 < lhs_dilation`.
* (C7) `size(rhs_dilation) = N - 2`.
* (C8) `0 < rhs_dilation`.
* (C9) `size(window_reversal) = N - 2`.
* (C10) `dim(lhs, input_batch_dimension) % batch_group_count = 0`.
* (C11) `dim(lhs, input_feature_dimension) % feature_group_count = 0`.
* (C12) `size(input_spatial_dimensions) = N - 2`.
* (C13) Given `input_dimensions = [input_batch_dimension] +
       input_spatial_dimensions + [input_feature_dimension]`:
  * `is_unique(input_dimensions)`.
  * `0 <= input_dimensions < N`.
* (C14) `dim(rhs, kernel_input_feature_dimension) = dim(lhs, input_feature_dimension) / feature_group_count`.
* (C15) `dim(rhs, kernel_output_feature_dimension) % batch_group_count = 0`.
* (C16) `dim(rhs, kernel_output_feature_dimension) % feature_group_count = 0`.
* (C17) `size(kernel_spatial_dimensions) = N - 2`.
* (C18) Given `kernel_dimensions = kernel_spatial_dimensions +
        [kernel_input_feature_dimension] + [kernel_output_feature_dimension]`:
  * `is_unique(kernel_dimensions)`.
  * `0 <= kernel_dimensions < N`.
* (C19) `size(output_spatial_dimensions) = N - 2`.
* (C20) Given `output_dimensions = [output_batch_dimension] +
        output_spatial_dimensions + [output_feature_dimension]`:
  * `is_unique(output_dimensions)`.
  * `0 <= output_dimensions < N`.
* (C21) `0 < feature_group_count`.
* (C22) `0 < batch_group_count`.
* (C23) `feature_group_count = 1 or batch_group_count = 1`.
* (C24) `size(precision_config) = 2`.
* (C25) `dim(result, result_dim)` is defined as:
  * `dim(lhs, input_batch_dimension) / batch_group_count` if `result_dim = output_batch_dimension`.
  * `dim(rhs, kernel_output_feature_dimension)` if `result_dim = output_feature_dimension`.
  * `num_windows` otherwise, where:
    * `output_spatial_dimensions[spatial_dim] = result_dim`.
    * `lhs_dim = input_spatial_dimensions[spatial_dim]`.
    * `rhs_dim = kernel_spatial_dimensions[spatial_dim]`.
    * `dilated_input_shape[lhs_dim] = dim(lhs, lhs_dim) = 0 ? 0 : (dim(lhs, lhs_dim) - 1) * lhs_dilation[spatial_dim] + 1`.
    * `padded_input_shape[lhs_dim] = padding[spatial_dim, 0] + dilated_input_shape[lhs_dim] + padding[spatial_dim, 1]`.
    * `dilated_window_shape[lhs_dim] = dim(rhs, rhs_dim) = 0 ? 0 : (dim(rhs, rhs_dim) - 1) * rhs_dilation[spatial_dim] + 1`.
    * `is_empty_window[lhs_dim] = padded_input_shape[lhs_dim] = 0 || dilated_window_shape[lhs_dim] > padded_input_shape[lhs_dim]`.
    * `num_windows = is_empty_window[lhs_dim] ? 0 : floor((padded_input_shape[lhs_dim] - dilated_window_shape[lhs_dim]) / window_strides[spatial_dim]) + 1`.
* (C26) `rank(result) = N`.
* If the operation uses non-quantized tensors:
  * (C27) `element_type(lhs) = element_type(rhs) = element_type(result)`.
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
<!-- markdownlint-enable line-length -->

#### Examples

```mlir
// %lhs: [[
//        [
//          [1], [2], [5], [6]
//        ],
//        [
//          [3], [4], [7], [8]
//        ],
//        [
//          [10], [11], [14], [15]
//        ],
//        [
//          [12], [13], [16], [17]
//        ]
//      ]]
//
// %rhs: [
//        [[[1]], [[1]], [[1]]],
//        [[[1]], [[1]], [[1]]],
//        [[[1]], [[1]], [[1]]]
//       ]
%result = "stablehlo.convolution"(%lhs, %rhs) {
  window_strides = array<i64: 4, 4>,
  padding = dense<0> : tensor<2x2xi64>,
  lhs_dilation = array<i64: 2, 2>,
  rhs_dilation = array<i64: 1, 1>,
  window_reversal = array<i1: false, false>,
  // In the StableHLO dialect, dimension numbers are encoded via:
  // `[<input dimensions>]x[<kernel dimensions>]->[output dimensions]`.
  // "b" is batch dimension, "f" is feature dimension,
  // "i" is input feature dimension, "o" is output feature dimension,
  // "0/1/etc" are spatial dimensions.
  dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
  batch_group_count = 1 : i64,
  feature_group_count = 1 : i64,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
} : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
// %result: [[
//            [[10], [26]],
//            [[46], [62]]
//          ]]
```

&nbsp;[More Examples](../stablehlo/tests/interpret/convolution.mlir)

### cosine

#### Semantics

Performs element-wise cosine operation on `operand` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `cos` from IEEE-754.
* For complex numbers: complex cosine.
* For quantized types: `dequantize_op_quantize(cosine, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [
//            [0.0, 1.57079632],       // [0, pi/2]
//            [3.14159265, 4.71238898] // [pi, 3pi/2]
//           ]
%result = "stablehlo.cosine"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[1.0, 0.0], [-1.0, 0.0]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/cosine.mlir)

### count_leading_zeros

#### Semantics

Performs element-wise count of the number of leading zero bits in the `operand`
tensor and produces a `result` tensor.

#### Inputs

| Label | Name      | Type                   | Constraints |
|-------|-----------|------------------------|-------------|
| (I1)  | `operand` | tensor of integer type | (C1)        |

#### Outputs

| Name     | Type                   | Constraints |
|----------|------------------------|-------------|
| `result` | tensor of integer type | (C1)        |

#### Constraints

* (C1) `type(operand) = type(result)`.

#### Examples

```mlir
// %operand: [[0, 1], [128, -1]]
%result = "stablehlo.count_leading_zeros"(%operand) : (tensor<2x2xi64>) -> tensor<2x2xi64>
// %result: [[64, 63], [56, 0]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/count_leading_zeros.mlir)

### custom_call

#### Semantics

Encapsulates an implementation-defined operation `call_target_name` that takes
`inputs` and `called_computations` and produces `results`. `has_side_effect`,
`backend_config` and `api_version` may be used to provide additional
implementation-defined metadata.

At the moment, this operation contains a fairly disorganized collection of
metadata which reflects organic evolution of its counterpart operation in
the XLA compiler. In the future, we are planning to unify this metadata
([#741](https://github.com/openxla/stablehlo/issues/741)).

#### Inputs

| Label | Name                  | Type                                              |
|-------|-----------------------|---------------------------------------------------|
| (I1)  | `inputs`              | variadic number of values                         |
| (I2)  | `call_target_name`    | constant of type `string`                         |
| (I3)  | `has_side_effect`     | constant of type `i1`                             |
| (I4)  | `backend_config`      | constant of type `string` or attribute dictionary |
| (I5)  | `api_version`         | constant of type `si32`                           |
| (I6)  | `called_computations` | variadic number of constants of type `string`     |

#### Outputs

| Name      | Type                      |
|-----------|---------------------------|
| `results` | variadic number of values |

#### Examples

```mlir
%results = "stablehlo.custom_call"(%input0) {
  call_target_name = "foo",
  has_side_effect = false,
  backend_config = {bar = 42 : i32},
  api_version = 4 : i32,
  called_computations = [@foo]
} : (tensor<f64>) -> tensor<f64>
```

### divide

#### Semantics

Performs element-wise division of dividend `lhs` and divisor `rhs` tensors and
produces a `result` tensor. Depending on the element type, does the following:

* For integers: integer division which produces the algebraic quotient with any
  fractional part discarded.
* For floats: `division` from IEEE-754.
* For complex numbers: complex division.
* For quantized types:
  * `dequantize_op_quantize(divide, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                                                                             | Constraints |
|-------|-------|----------------------------------------------------------------------------------|-------------|
| (I1)  | `lhs` | tensor of integer, floating-point or complex type or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor of integer, floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                              | Constraints |
|----------|-----------------------------------------------------------------------------------|-------------|
| `result` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(lhs) = baseline_type(rhs) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [17.1, -17.1, 17.1, -17.1]
// %rhs: [3.0, 3.0, -3.0, -3.0]
%result = "stablehlo.divide"(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// %result: [5.66666651, -5.66666651, -5.66666651, 5.66666651]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/divide.mlir)

### dot_general

#### Semantics

Computes dot products between slices of `lhs` and slices of `rhs` and produces a
`result` tensor.

More formally, `result[result_index] = dot_product`, where:

<!-- markdownlint-disable line-length -->
* `lhs_result_dimensions = [d for d in axes(lhs) and d not in lhs_batching_dimensions and d not in lhs_contracting_dimensions]`.
* `rhs_result_dimensions = [d for d in axes(rhs) and d not in rhs_batching_dimensions and d not in rhs_contracting_dimensions]`.
* `result_batching_index + result_lhs_index + result_rhs_index = result_index`
  where `size(result_batching_index) = size(lhs_batching_dimensions)`,
  `size(result_lhs_index) = size(lhs_result_dimensions)` and
  `size(result_rhs_index) = size(rhs_result_dimensions)`.
* `transposed_lhs = transpose(lhs, lhs_batching_dimensions + lhs_result_dimensions + lhs_contracting_dimensions)`.
* `transposed_lhs_slice = slice(transposed_lhs, result_batching_index + result_lhs_index + [:, ..., :])`.
* `reshaped_lhs_slice = reshape(transposed_lhs_slice, dims(lhs, lhs_contracting_dimensions))`.
* `transposed_rhs = transpose(rhs, rhs_batching_dimensions + rhs_result_dimensions + rhs_contracting_dimensions)`.
* `transposed_rhs_slice = slice(transposed_rhs, result_batching_index + result_rhs_index + [:, ..., :])`.
* `reshaped_rhs_slice = reshape(transposed_rhs_slice, dims(rhs, rhs_contracting_dimensions))`.
* `dot_product = reduce(
    inputs=[multiply(reshaped_lhs_slice, reshaped_rhs_slice)],
    init_values=[constant(0, element_type(result))],
    dimensions=range(size(lhs_contracting_dimensions)),
    body=lambda x, y: add(x, y))`.
<!-- markdownlint-enable line-length -->

For quantized types, performs `dequantize_op_quantize(
    lambda lhs, rhs: dot_general(lhs, rhs, lhs_batching_dimensions,
        rhs_batching_dimensions, lhs_contracting_dimensions,
        rhs_contracting_dimensions, precision_config), lhs, rhs, type(result))`.

For hybrid quantized types, performs `hybrid_dequantize_then_op(
    lambda lhs, rhs: dot_general(lhs, rhs, lhs_batching_dimensions,
        rhs_batching_dimensions, lhs_contracting_dimensions,
        rhs_contracting_dimensions, precision_config), lhs, rhs)`.

`precision_config` controls the tradeoff between speed and accuracy for
computations on accelerator backends. This can be one of the following (at the
moment, the semantics of these enum values is underspecified, but we are
planning to address this in
[#755](https://github.com/openxla/stablehlo/issues/755)):

* `DEFAULT`: Fastest calculation, but least accurate approximation to the
  original number.
* `HIGH`: Slower calculation, but more accurate approximation to the
  original number.
* `HIGHEST`: Slowest calculation, but most accurate approximation to the
  original number.

A `DotAlgorithm` defines the main properties of the algorithm used to implement
the dot operation, which also defines the precision. If the algorithm attribute
fields are set, then the `precision_config` must be `DEFAULT`. `DotAlgorithms`
do not have a default value, as the default parameters are implementation
defined. As such, all dot algorithm fields may be set to `None` to specify an
empty dot algorithm, which will instead use the `precision_config` value.

`DotAlgorithm` fields include:

* `lhs_precision_type` and `rhs_precision_type`, the precisions that the LHS and
  RHS of the operation are rounded to. Precision types are independent from the
  storage types of the inputs and the output.
* `accumulation_type` the precision used for accumulation.
* `lhs_component_count`, `rhs_component_count`, and `num_primitive_operations`
  apply when we are doing an algorithm which decomposes the LHS and/or RHS into
  multiple components and does multiple "primitive" dot operations on those
  values - usually to emulate a higher precision (e.g.
[Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376.pdf):
  bf16_6x tf32_3x, etc). For algorithms with no decomposition, these values
  should be set to `1`.
* `allow_imprecise_accumulation` to specify if accumulation in lower precision
  is permitted for some steps (e.g. `CUBLASLT_MATMUL_DESC_FAST_ACCUM`).

Example `DotAlgorithm` attributes:

```txt
// Inputs are casted to tf32, and then accumulated in f32:
{lhs_precision_type = tf32,
 rhs_precision_type = tf32,
 accumulation_type = f32,
 lhs_component_count = 1,
 rhs_component_count = 1,
 num_primitive_operations = 1,
 allow_imprecise_accumulation = false}


// bf16_6x: each input is decomposed to 3 bf16 components, then 6 dot operations are done on those components, and the result is accumulated in f32.
{lhs_precision_type = bf16,
 rhs_precision_type = bf16,
 accumulation_type = f32,
 lhs_component_count = 3,
 rhs_component_count = 3,
 num_primitive_operations = 6,
 allow_imprecise_accumulation = false}


// Inputs are (casted to) f8e5m2, and we accumulate in f32, but for some steps we may accumulate in lower precision.
{lhs_precision_type = f8e5m2,
 rhs_precision_type = f8e5m2,
 accumulation_type = f32,
 lhs_component_count = 1,
 rhs_component_count = 1,
 num_primitive_operations = 1,
 allow_imprecise_accumulation = true}
```

It is up to the implementations to decide which combinations are supported. In
general, it is not guaranteed that each algorithm is supported on each
accelerator type by the consumer of the StableHLO. If a given algorithm is not
supported, an error should be raised as opposed to falling back to an
alternative. StableHLO verification will provide best effort verification,
preventing algorithms that are not known to be supported on *any* hardware.

See [`xla_data.proto > Algorithm`](https://github.com/openxla/xla/blob/e8a707554de6b3d6bfd891583a81ff7020a97b54/xla/xla_data.proto#L1022)
for some supported algorithm values. Ticket #2483 captures the plan to create a
centralized doc on supported algorithms by backend.

#### Inputs

| Label | Name                           | Type                                                         | Constraints                                    |
|-------|--------------------------------|--------------------------------------------------------------|------------------------------------------------|
| (I1)  | `lhs`                          | tensor or per-tensor quantized tensor                        | (C5-C6), (C9-C10), (C12-C14), (C17-C18), (C20) |
| (I2)  | `rhs`                          | tensor or quantized tensor                                   | (C7-C10), (C12-C20)                            |
| (I3)  | `lhs_batching_dimensions`      | 1-dimensional tensor constant of type `si64`                 | (C1), (C3), (C5), (C9), (C12)                  |
| (I4)  | `rhs_batching_dimensions`      | 1-dimensional tensor constant of type `si64`                 | (C1), (C4), (C7), (C9)                         |
| (I5)  | `lhs_contracting_dimensions`   | 1-dimensional tensor constant of type `si64`                 | (C2), (C3), (C6), (C10)                        |
| (I6)  | `rhs_contracting_dimensions`   | 1-dimensional tensor constant of type `si64`                 | (C2), (C4), (C8), (C10), (C16)                 |
| (I7)  | `precision_config`             | variadic number of enums of `DEFAULT`, `HIGH`, and `HIGHEST` | (C11), (C21)                                   |
| (I8)  | `lhs_precision_type`           | FloatType or TensorFloat32                                   | (C21)                                          |
| (I9)  | `rhs_precision_type`           | FloatType or TensorFloat32                                   | (C21)                                          |
| (I10) | `accumulation_type`            | FloatType or TensorFloat32                                   | (C21)                                          |
| (I11) | `lhs_component_count`          | constant of type `si32`                                      | (C21), (C22)                                   |
| (I12) | `rhs_component_count`          | constant of type `si32`                                      | (C21), (C23)                                   |
| (I13) | `num_primitive_operations`     | constant of type `si32`                                      | (C21), (C24)                                   |
| (I14) | `allow_imprecise_accumulation` | constant of type `bool`                                      | (C21)                                          |

#### Outputs

| Name     | Type                       | Constraints             |
|----------|----------------------------|-------------------------|
| `result` | tensor or quantized tensor | (C12), (C14), (C18-C20) |

#### Constraints

* (C1) `size(lhs_batching_dimensions) = size(rhs_batching_dimensions)`.
* (C2) `size(lhs_contracting_dimensions) =
  size(rhs_contracting_dimensions)`.
* (C3) `is_unique(lhs_batching_dimensions + lhs_contracting_dimensions)`.
* (C4) `is_unique(rhs_batching_dimensions + rhs_contracting_dimensions)`.
* (C5) `0 <= lhs_batching_dimensions < rank(lhs)`.
* (C6) `0 <= lhs_contracting_dimensions < rank(lhs)`.
* (C7) `0 <= rhs_batching_dimensions < rank(rhs)`.
* (C8) `0 <= rhs_contracting_dimensions < rank(rhs)`.
* (C9) `dim(lhs, lhs_batching_dimensions...) =
  dim(rhs, rhs_batching_dimensions...)`.
* (C10) `dim(lhs, lhs_contracting_dimensions...) =
  dim(rhs, rhs_contracting_dimensions...)`.
* (C11) `size(precision_config) = 2`.
* (C12) `shape(result) = dim(lhs, lhs_batching_dimensions) +
  dim(lhs, lhs_result_dimensions) + dim(rhs, rhs_result_dimensions)`.
* If the operation uses non-quantized tensors:
  * (C13) `element_type(lhs) = element_type(rhs)`.
* If the operation uses quantized tensors:
  * (C14) `is_quantized(lhs) = is_quantized(result) and is_quantized(rhs)`.
  * (C15) `zero_points(rhs) = 0`.
  * (C16) If `is_per_axis_quantized(rhs)`, then
    `quantization_dimension(rhs)` not in `rhs_contracting_dimensions`.
  * If `is_quantized(lhs)`:
    * (C17) `storage_type(lhs) = storage_type(rhs)`.
    * (C18) `expressed_type(lhs) = expressed_type(rhs) = expressed_type(result)`.
    * (C19) If `is_per_tensor_quantized(rhs)`, then
      `is_per_tensor_quantized(result)`.
  * If `!is_quantized(lhs)`:
    * (C20) `element_type(lhs) = expressed_type(rhs) = element_type(result)`.
* If `!is_empty_algorithm(lhs_precision_type, rhs_precision_type,
  accumulation_type, lhs_component_count, rhs_component_count,
  num_primitive_operations allow_imprecise_accumulation)`:
  * (C21) `precision_config... = DEFAULT`.
  * (C22) `0 < lhs_component_count`.
  * (C23) `0 < rhs_component_count`.
  * (C24) `0 < num_primitive_operations`.

#### Examples

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
    lhs_precision_type = tf32,
    rhs_precision_type = tf32,
    accumulation_type = f32,
    lhs_component_count = 1,
    rhs_component_count = 1,
    num_primitive_operations = 1,
    allow_imprecise_accumulation = false
  >
} : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
// %result: [
//           [[1, 2],
//            [3, 4]],
//           [[5, 6],
//            [7, 8]]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dot_general.mlir)

### dynamic_broadcast_in_dim

#### Semantics

This operation is functionally identical to
[broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim)
op, but the result shape is specified dynamically via `output_dimensions`.

The operation also accepts optional attributes `known_expanding_dimensions`, `known_nonexpanding_dimensions`
to express static knowledge about the expanding behavior of dimensions.
If not specified, all dimensions are assumed to be possibly expanding.

#### Inputs

| Label | Name                             | Type                                          | Constraints            |
|-------|----------------------------------|-----------------------------------------------|------------------------|
| (I1)  | `operand`                        | tensor or quantized tensor                    | (C1-C2), (C5-C6), (C9) |
| (I2)  | `output_dimensions`              | 1-dimensional tensor of integer type          | (C7)                   |
| (I3)  | `broadcast_dimensions`           | 1-dimensional constant tensor of integer type | (C2-C6)                |
| (I4)  | `known_expanding_dimensions`     | 1-dimensional constant tensor of integer type | (C8-C9)                |
| (I5)  | `known_nonexpanding_dimensions` | 1-dimensional constant tensor of integer type | (C8-C9)                |

#### Outputs

| Name     | Type                       | Constraints         |
|----------|----------------------------|---------------------|
| `result` | tensor or quantized tensor | (C1), (C3), (C5-C7) |

#### Constraints

* (C1) `element_type(result)` is given by:
  * `element_type(operand)`, if `!is_per_axis_quantized(operand)`.
  * `element_type(operand)` except that `quantization_dimension(operand)`,
  `scales(operand)`, and `zero_points(operand)` may differ from
  `quantization_dimension(result)`, `scales(result)`, and `zero_points(result)`
  resp., otherwise.
* (C2) `size(broadcast_dimensions) = rank(operand)`.
* (C3) `0 <= broadcast_dimensions < rank(result)`.
* (C4) `is_unique(broadcast_dimensions)`.
* (C5) For all `d` in `axes(operand)`:
  * `dim(operand, d) = 1` or
  * `dim(operand, d) = dim(result, broadcast_dimensions[d])`.
* (C6) If `is_per_axis_quantized(result)`:
  * `quantization_dimension(result) = broadcast_dimensions[quantization_dimension(operand)]`.
  * If `dim(operand, quantization_dimension(operand)) = 1`, then
    `scales(result)[i] = scales(operand)[0] and zero_points(result)[i] =
    zero_points(operand)[0] for i in
    range(dim(result, quantization_dimension(result)))`.
* (C7) `size(output_dimensions) = rank(result)`.
* (C8) `is_unique(known_expanding_dimensions + known_nonexpanding_dimensions)`.
* (C9) `0 <= known_expanding_dimensions < rank(operand)`.
* (C10) `0 <= known_nonexpanding_dimensions < rank(operand)`.

#### Examples

```mlir
// %operand: [
//            [1, 2, 3]
//           ]
%operand = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
%output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
%result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
  broadcast_dimensions = array<i64: 2, 1>,
  known_expanding_dimensions = array<i64: 0>,
  known_nonexpanding_dimensions = array<i64: 1>
} : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
// %result: [
//            [
//             [1, 1],
//             [2, 2],
//             [3, 3]
//            ],
//            [
//             [1, 1],
//             [2, 2],
//             [3, 3]
//            ]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_broadcast_in_dim.mlir)

### dynamic_conv

#### Semantics

This operation is functionally identical to
[convolution](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution)
op, but the padding is specified dynamically via `padding`.

#### Inputs

| Label | Name                              | Type                                                         | Constraints                                               |
|-------|-----------------------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| (I1)  | `lhs`                             | tensor or per-tensor quantized tensor                        | (C1), (C10-C11), (C14) (C25), (C26-C27), (C30-C31), (C33) |
| (I2)  | `rhs`                             | tensor or quantized tensor                                   | (C1), (C14-C16), (C26-C28), (C30-C33)                     |
| (I3)  | `padding`                         | 2-dimensional tensor of integer type                         | (C4)                                                      |
| (I4)  | `window_strides`                  | 1-dimensional tensor constant of type `si64`                 | (C2-C3)                                                   |
| (I5)  | `lhs_dilation`                    | 1-dimensional tensor constant of type `si64`                 | (C5-C6)                                                   |
| (I6)  | `rhs_dilation`                    | 1-dimensional tensor constant of type `si64`                 | (C7-C8)                                                   |
| (I7)  | `window_reversal`                 | 1-dimensional tensor constant of type `i1`                   | (C9)                                                      |
| (I8)  | `input_batch_dimension`           | constant of type `si64`                                      | (C10), (C13)                                              |
| (I9)  | `input_feature_dimension`         | constant of type `si64`                                      | (C11), (C13-C14)                                          |
| (I10) | `input_spatial_dimensions`        | 1-dimensional tensor constant of type `si64`                 | (C12), (C13)                                              |
| (I11) | `kernel_input_feature_dimension`  | constant of type `si64`                                      | (C14), (C18)                                              |
| (I12) | `kernel_output_feature_dimension` | constant of type `si64`                                      | (C15-C16), (C18), (C28)                                   |
| (I13) | `kernel_spatial_dimensions`       | 1-dimensional tensor constant of type `si64`                 | (C17-C18)                                                 |
| (I14) | `output_batch_dimension`          | constant of type `si64`                                      | (C20)                                                     |
| (I15) | `output_feature_dimension`        | constant of type `si64`                                      | (C20), (C29)                                              |
| (I16) | `output_spatial_dimensions`       | 1-dimensional tensor constant of type `si64`                 | (C19-C20)                                                 |
| (I17) | `feature_group_count`             | constant of type `si64`                                      | (C11), (C14), (C16), (C21), (C23)                         |
| (I18) | `batch_group_count`               | constant of type `si64`                                      | (C10), (C15), (C22), (C23)                                |
| (I19) | `precision_config`                | variadic number of enums of `DEFAULT`, `HIGH`, and `HIGHEST` | (C24)                                                     |

#### Outputs

| Name     | Type                       | Constraints                 |
|----------|----------------------------|-----------------------------|
| `result` | tensor or quantized tensor | (C25-C27), (C29), (C31-C33) |

#### Constraints

<!-- markdownlint-disable line-length -->
* (C1) `N = rank(lhs) = rank(rhs)`.
* (C2) `size(window_strides) = N - 2`.
* (C3) `0 < window_strides`.
* (C4) `shape(padding) = [N - 2, 2]`.
* (C5) `size(lhs_dilation) = N - 2`.
* (C6) `0 < lhs_dilation`.
* (C7) `size(rhs_dilation) = N - 2`.
* (C8) `0 < rhs_dilation`.
* (C9) `size(window_reversal) = N - 2`.
* (C10) `dim(lhs, input_batch_dimension) % batch_group_count = 0`.
* (C11) `dim(lhs, input_feature_dimension) % feature_group_count = 0`.
* (C12) `size(input_spatial_dimensions) = N - 2`.
* (C13) Given `input_dimensions = [input_batch_dimension] +
       input_spatial_dimensions + [input_feature_dimension]`:
  * `is_unique(input_dimensions)`.
  * `0 <= input_dimensions < N`.
* (C14) `dim(rhs, kernel_input_feature_dimension) = dim(lhs, input_feature_dimension) / feature_group_count`.
* (C15) `dim(rhs, kernel_output_feature_dimension) % batch_group_count = 0`.
* (C16) `dim(rhs, kernel_output_feature_dimension) % feature_group_count = 0`.
* (C17) `size(kernel_spatial_dimensions) = N - 2`.
* (C18) Given `kernel_dimensions = kernel_spatial_dimensions +
        [kernel_input_feature_dimension] + [kernel_output_feature_dimension]`:
  * `is_unique(kernel_dimensions)`.
  * `0 <= kernel_dimensions < N`.
* (C19) `size(output_spatial_dimensions) = N - 2`.
* (C20) Given `output_dimensions = [output_batch_dimension] +
        output_spatial_dimensions + [output_feature_dimension]`:
  * `is_unique(output_dimensions)`.
  * `0 <= output_dimensions < N`.
* (C21) `0 < feature_group_count`.
* (C22) `0 < batch_group_count`.
* (C23) `feature_group_count = 1 or batch_group_count = 1`.
* (C24) `size(precision_config) = 2`.
* (C25) `dim(result, result_dim)` is defined as:
  * `dim(lhs, input_batch_dimension) / batch_group_count` if `result_dim = output_batch_dimension`.
  * `dim(rhs, kernel_output_feature_dimension)` if `result_dim = output_feature_dimension`.
  * `num_windows` otherwise, where:
    * `output_spatial_dimensions[spatial_dim] = result_dim`.
    * `lhs_dim = input_spatial_dimensions[spatial_dim]`.
    * `rhs_dim = kernel_spatial_dimensions[spatial_dim]`.
    * `dilated_input_shape[lhs_dim] = dim(lhs, lhs_dim) = 0 ? 0 : (dim(lhs, lhs_dim) - 1) * lhs_dilation[spatial_dim] + 1`.
    * `padded_input_shape[lhs_dim] = padding[spatial_dim, 0] + dilated_input_shape[lhs_dim] + padding[spatial_dim, 1]`.
    * `dilated_window_shape[lhs_dim] = dim(rhs, rhs_dim) = 0 ? 0 : (dim(rhs, rhs_dim) - 1) * rhs_dilation[spatial_dim] + 1`.
    * `is_empty_window[lhs_dim] = padded_input_shape[lhs_dim] = 0 || dilated_window_shape[lhs_dim] > padded_input_shape[lhs_dim]`.
    * `num_windows = is_empty_window[lhs_dim] ? 0 : floor((padded_input_shape[lhs_dim] - dilated_window_shape[lhs_dim]) / window_strides[spatial_dim]) + 1`.
* (C26) `rank(result) = N`.
* If the operation uses non-quantized tensors:
  * (C27) `element_type(lhs) = element_type(rhs) = element_type(result)`.
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
<!-- markdownlint-enable line-length -->

#### Examples

```mlir
// %lhs: [[
//        [[1], [2], [5], [6]],
//        [[3], [4], [7], [8]],
//        [[10], [11], [14], [15]],
//        [[12], [13], [16], [17]]
//      ]]
//
// %rhs: [
//         [[[1]], [[1]], [[1]]],
//         [[[1]], [[1]], [[1]]],
//         [[[1]], [[1]], [[1]]]
//        ]
// %padding: [[1, 1],
//            [1, 1]]
%result = "stablehlo.dynamic_conv"(%lhs, %rhs, %padding) {
  window_strides = array<i64: 4, 4>,
  lhs_dilation = array<i64: 2, 2>,
  rhs_dilation = array<i64: 1, 1>,
  window_reversal = array<i1: false, false>,
  dimension_numbers = #stablehlo.conv<raw
    input_batch_dimension = 0,
    input_feature_dimension = 3,
    input_spatial_dimensions = [0, 1],
    kernel_input_feature_dimension = 2,
    kernel_output_feature_dimension = 3,
    kernel_spatial_dimensions = [0, 1],
    output_batch_dimension = 0,
    output_feature_dimension = 3,
    output_spatial_dimensions = [1, 2]
  >,
  feature_group_count = 1 : i64,
  batch_group_count = 1 : i64,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
} : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>, tensor<2x2xi64>) -> tensor<1x2x2x1xi64>
// %result: [[
//            [[1], [5]],
//            [[10], [14]]
//          ]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_conv.mlir)

### dynamic_gather

#### Semantics

This operation is functionally identical to
[gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather)
op, with the `slice_sizes` specified dynamically as a value.

#### Inputs

| Label | Name                   | Type                                         | Constraints                  |
|-------|------------------------|----------------------------------------------|------------------------------|
| (I1)  | `operand`              | tensor or per-tensor quantized tensor        | (C1), (C7), (C10-C12), (C14) |
| (I2)  | `start_indices`        | tensor of integer type                       | (C2), (C3), (C13)            |
| (I3)  | `slice_sizes`          | 1-dimensional tensor of integer type         | (C8), (C11-C13)              |
| (I4)  | `offset_dims`          | 1-dimensional tensor constant of type `si64` | (C1), (C4-C5), (C13)         |
| (I5)  | `collapsed_slice_dims` | 1-dimensional tensor constant of type `si64` | (C1), (C6-C8), (C13)         |
| (I6)  | `start_index_map`      | 1-dimensional tensor constant of type `si64` | (C3), (C9), (C10)            |
| (I7)  | `index_vector_dim`     | constant of type `si64`                      | (C2), (C3), (C13)            |
| (I8)  | `indices_are_sorted`   | constant of type `i1`                        |                              |

#### Outputs

| Name     | Type                                  | Constraints     |
|----------|---------------------------------------|-----------------|
| `result` | tensor or per-tensor quantized tensor | (C5), (C13-C14) |

#### Constraints

* (C1) `rank(operand) = size(offset_dims) + size(collapsed_slice_dims)`.
* (C2) `0 <= index_vector_dim <= rank(start_indices)`.
* (C3) `size(start_index_map) =
       index_vector_dim < rank(start_indices) ?
       dim(start_indices, index_vector_dim) : 1`.
* (C4) `is_unique(offset_dims) and is_sorted(offset_dims)`.
* (C5) `0 <= offset_dims < rank(result)`.
* (C6) `is_unique(collapsed_slice_dims) and is_sorted(collapsed_slice_dims)`.
* (C7) `0 <= collapsed_slice_dims < rank(operand)`.
* (C8) `slice_sizes[collapsed_slice_dims...] <= 1`.
* (C9) `is_unique(start_index_map)`.
* (C10) `0 <= start_index_map < rank(operand)`.
* (C11) `size(slice_sizes) = rank(operand)`.
* (C12) `0 <= slice_sizes <= shape(operand)`.
* (C13) `shape(result) = combine(batch_dim_sizes, offset_dim_sizes)` where:
  * `batch_dim_sizes = shape(start_indices)` except that the dimension size
    of `start_indices` corresponding to `index_vector_dim` is not included.
  * `offset_dim_sizes = shape(slice_sizes)` except that the dimension sizes
    in `slice_sizes` corresponding to `collapsed_slice_dims` are not included.
  * `combine` puts `batch_dim_sizes` at axes corresponding to `batch_dims` and
   `offset_dim_sizes` at axes corresponding to `offset_dims`.
* (C14) `element_type(operand) = element_type(result)`.

#### Examples

```mlir
// %operand: [
//            [[1, 2], [3, 4], [5, 6], [7, 8]],
//            [[9, 10],[11, 12], [13, 14], [15, 16]],
//            [[17, 18], [19, 20], [21, 22], [23, 24]]
//           ]
// %start_indices: [
//                  [[0, 0], [1, 0], [2, 1]],
//                  [[0, 1], [1, 1], [0, 2]]
//                 ]
// %slize_sizes: [1, 2, 2]
%result = "stablehlo.dynamic_gather"(%operand, %start_indices, %slize_sizes) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [2, 3],
    collapsed_slice_dims = [0],
    start_index_map = [1, 0],
    index_vector_dim = 2>,
  indices_are_sorted = false
} : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi64>
// %result: [
//            [
//              [[1, 2], [3, 4]],
//              [[3, 4], [5, 6]],
//              [[13, 14], [15, 16]]
//            ],
//            [
//              [[9, 10], [11, 12]],
//              [[11, 12], [13, 14]],
//              [[17, 18], [19, 20]]
//            ]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_gather.mlir)

### dynamic_iota

#### Semantics

This operation is functionally identical to
[iota](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota)
op, but the result shape is specified dynamically via `output_shape`.

#### Inputs

| Label | Name             | Type                                 | Constraints |
|-------|------------------|--------------------------------------|-------------|
| (I1)  | `output_shape`   | 1-dimensional tensor of integer type | (C1), (C2)  |
| (I2)  | `iota_dimension` | `si64`                               | (C1)        |

#### Outputs

| Name     | Type                                                                              | Constraints |
|----------|-----------------------------------------------------------------------------------|-------------|
| `result` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C2)        |

#### Constraints

* (C1) `0 <= iota_dimension < size(output_shape)`.
* (C2) `rank(result) = size(output_shape)`.

#### Examples

```mlir
%output_shape = stablehlo.constant dense<[4, 5]> : tensor<2xi64>
%result = "stablehlo.dynamic_iota"(%output_shape) {
  iota_dimension = 0 : i64
} : (tensor<2xi64>) -> tensor<4x5xi64>
// %result: [
//           [0, 0, 0, 0, 0],
//           [1, 1, 1, 1, 1],
//           [2, 2, 2, 2, 2],
//           [3, 3, 3, 3, 3]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_iota.mlir)

### dynamic_pad

#### Semantics

This operation is functionally identical to
[pad](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad)
op, but with `edge_padding_low`, `edge_padding_high`, and `interior_padding`
specified dynamically as values.

#### Inputs

| Label | Name                | Type                                                | Constraints      |
|-------|---------------------|-----------------------------------------------------|------------------|
| (I1)  | `operand`           | tensor or per-tensor quantized tensor               | (C1), (C2), (C4) |
| (I2)  | `padding_value`     | 0-dimensional tensor or per-tensor quantized tensor | (C1)             |
| (I3)  | `edge_padding_low`  | 1-dimensional tensor of integer type                | (C1), (C4)       |
| (I4)  | `edge_padding_high` | 1-dimensional tensor of integer type                | (C1), (C4)       |
| (I5)  | `interior_padding`  | 1-dimensional tensor of integer type                | (C2-C4)          |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C3-C6)     |

#### Constraints

* (C1) `element_type(operand) = element_type(padding_value) =
  element_type(result)`.
* (C2) `size(edge_padding_low) = size(edge_padding_high) =
  size(interior_padding) = rank(operand)`.
* (C3) `0 <= interior_padding`.
* (C4) `shape(result) = shape(operand) + edge_padding_low +
  max(shape(operand) - 1, 0) * interior_padding + edge_padding_high`.

#### Examples

```mlir
// %operand: [
//            [1, 2, 3],
//            [4, 5, 6]
//           ]
// %padding_value: 0
// %edge_padding_low: [0, 1]
// %edge_padding_high: [2, 1]
// %interior_padding: [1, 2]
%result = "stablehlo.dynamic_pad"(%operand, %padding_value,
  %edge_padding_low, %edge_padding_high, %interior_padding
) : (tensor<2x3xi64>, tensor<i64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x9xi64>
// %result: [
//           [0, 1, 0, 0, 2, 0, 0, 3, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 4, 0, 0, 5, 0, 0, 6, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_pad.mlir)

### dynamic_reshape

#### Semantics

This operation is functionally identical to
[reshape](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape)
op, but the result shape is specified dynamically via `output_shape`.

#### Inputs

| Label | Name           | Type                                 | Constraints |
|-------|----------------|--------------------------------------|-------------|
| (I1)  | `operand`      | tensor or quantized tensor           | (C1-C3)     |
| (I2)  | `output_shape` | 1-dimensional tensor of integer type | (C4)        |

#### Outputs

| Name     | Type                       | Constraints |
|----------|----------------------------|-------------|
| `result` | tensor or quantized tensor | (C1-C4)     |

#### Constraints

* (C1) `element_type(result)` is given by:
  * `element_type(operand)`, if `!is_per_axis_quantized(operand)`.
  * `element_type(operand)` except that `quantization_dimension(operand)` and
    `quantization_dimension(result)` may differ, otherwise.
* (C2) `size(operand) = size(result)`.
* (C3) If `is_per_axis_quantized(operand)`:
  * `reduce(dims(operand, [0, 1, ..., quantization_dimension(operand) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y) =
    reduce(dims(result, [0, 1, ..., quantization_dimension(result) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y)`.
  * `dim(operand, quantization_dimension(operand)) =
    dim(result, quantization_dimension(result))`.
  * `reduce(dims(operand,
    [quantization_dimension(operand) + 1, ..., rank(operand) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y) =
    reduce(dims(result,
    [quantization_dimension(result) + 1, ..., rank(result) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y)`.
* (C4) `size(output_shape) = rank(result)`.

#### Examples

```mlir
// %operand: [[1, 2, 3], [4, 5, 6]]
// %output_shape: [3, 2]
%result = "stablehlo.dynamic_reshape"(%operand, %output_shape) : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<3x2xi64>
// %result: [[1, 2], [3, 4], [5, 6]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_reshape.mlir)

### dynamic_slice

#### Semantics

Extracts a slice from the `operand` using dynamically-computed starting indices
and produces a `result` tensor. `start_indices` contain the starting indices of
the slice for each dimension subject to potential adjustment, and `slice_sizes`
contain the sizes of the slice for each dimension. More formally,
`result[result_index] = operand[operand_index]` where:

* `adjusted_start_indices = clamp(0, start_indices, shape(operand) -
  slice_sizes)`.
* `operand_index = adjusted_start_indices + result_index`.

#### Inputs

| Label | Name            | Type                                                     | Constraints      |
|-------|-----------------|----------------------------------------------------------|------------------|
| (I1)  | `operand`       | tensor or per-tensor quantized tensor                    | (C1), (C2), (C4) |
| (I2)  | `start_indices` | variadic number of 0-dimensional tensors of integer type | (C2), (C3)       |
| (I3)  | `slice_sizes`   | 1-dimensional tensor constant of type `si64`             | (C2), (C4), (C5) |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1), (C5)  |

#### Constraints

* (C1) `element_type(operand) = element_type(result)`.
* (C2) `size(start_indices) = size(slice_sizes) = rank(operand)`.
* (C3) `same(type(start_indices...))`.
* (C4) `0 <= slice_sizes <= shape(operand)`.
* (C5) `shape(result) = slice_sizes`.

#### Examples

```mlir
// %operand: [
//            [0, 0, 1, 1],
//            [0, 0, 1, 1],
//            [0, 0, 0, 0],
//            [0, 0, 0, 0]
//           ]
// %start_indices0: -1
// %start_indices1: 3
%result = "stablehlo.dynamic_slice"(%operand, %start_indices0, %start_indices1) {
  slice_sizes = array<i64: 2, 2>
} : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
// %result: [
//           [1, 1],
//           [1, 1]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_slice.mlir)

### dynamic_update_slice

#### Semantics

Produces a `result` tensor which is equal to the `operand` tensor except that
the slice starting at `start_indices` is updated with the values in `update`.
More formally, `result[result_index]` is defined as:

* `update[update_index]` if `0 <= update_index < shape(update)` where:
  * `adjusted_start_indices = clamp(0, start_indices, shape(operand) -
    shape(update))`.
  * `update_index = result_index - adjusted_start_indices`.
* `operand[result_index]` otherwise.

#### Inputs

| Label | Name            | Type                                                     | Constraints      |
|-------|-----------------|----------------------------------------------------------|------------------|
| (I1)  | `operand`       | tensor or per-tensor quantized tensor                    | (C1-C4), (C6)    |
| (I2)  | `update`        | tensor or per-tensor quantized tensor                    | (C2), (C3), (C6) |
| (I3)  | `start_indices` | variadic number of 0-dimensional tensors of integer type | (C4), (C5)       |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `type(operand) = type(result)`.
* (C2) `element_type(update) = element_type(operand)`.
* (C3) `rank(update) = rank(operand)`.
* (C4) `size(start_indices) = rank(operand)`.
* (C5) `same(type(start_indices...))`.
* (C6) `0 <= shape(update) <= shape(operand)`.

#### Examples

```mlir
// %operand: [
//            [1, 1, 0, 0],
//            [1, 1, 0, 0],
//            [1, 1, 1, 1],
//            [1, 1, 1, 1]
//           ]
// %update: [
//           [1, 1],
//           [1, 1]
//          ]
// %start_indices0: -1
// %start_indices1: 3
%result = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1)
  : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
// %result: [
//           [1, 1, 1, 1],
//           [1, 1, 1, 1],
//           [1, 1, 1, 1],
//           [1, 1, 1, 1]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/dynamic_update_slice.mlir)

### exponential

#### Semantics

Performs element-wise exponential operation on `operand` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `exp` from IEEE-754.
* For complex numbers: complex exponential.
* For quantized types:
  `dequantize_op_quantize(exponential, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [[0.0, 1.0], [2.0, 3.0]]
%result = "stablehlo.exponential"(%operand) : (tensor<2x2xf64>) -> tensor<2x2xf64>
// %result: [[1.0, 2.7182818284590451], [7.3890560989306504, 20.085536923187668]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/exponential.mlir)

### exponential_minus_one

#### Semantics

Performs element-wise exponential minus one operation on `operand` tensor and
produces a `result` tensor. Depending on the element type, does the following:

* For floats: `expm1` from IEEE-754.
* For complex numbers: complex exponential minus one.
* For quantized types:
  `dequantize_op_quantize(exponential_minus_one, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [0.0, 1.0]
%result = "stablehlo.exponential_minus_one"(%operand) : (tensor<2xf64>) -> tensor<2xf64>
// %result: [0.0, 1.71828187]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/exponential_minus_one.mlir)

### fft

#### Semantics

Performs the forward and inverse Fourier transforms for real and complex
inputs/outputs.

`fft_type` is one of the following:

* `FFT`: Forward complex-to-complex FFT.
* `IFFT`: Inverse complex-to-complex FFT.
* `RFFT`: Forward real-to-complex FFT.
* `IRFFT`: Inverse real-to-complex FFT (i.e. takes complex, returns real).

More formally, given the function `fft` which takes 1-dimensional tensors of
complex types as input, produces 1-dimensional tensors of same types as
output and computes the discrete Fourier transform:

For `fft_type = FFT`, `result` is defined as the final result of a series of L
computations where `L = size(fft_length)`. For example, for `L = 3`:

* `result1[i0, ..., :] = fft(operand[i0, ..., :])`.
* `result2[i0, ..., :, iR-1] = fft(result1[i0, ..., :, iR-1])`.
* `result[i0, ..., :, iR-2, iR-1] = fft(result2[i0, ..., :, iR-2, iR-1])`.

Furthermore, given the function `ifft` which has the same type signature and
computes the inverse of `fft`:

For `fft_type = IFFT`, `result` is defined as the inverse of the computations
for `fft_type = FFT`. For example, for `L = 3`:

* `result1[i0, ..., :, iR-2, iR-1] = ifft(operand[i0, ..., :, iR-2, iR-1])`.
* `result2[i0, ..., :, iR-1] = ifft(result1[i0, ..., :, iR-1])`.
* `result[i0, ..., :] = ifft(result2[i0, ..., :])`.

Furthermore, given the function `rfft` which takes 1-dimensional tensors of
floating-point types, produces 1-dimensional tensors of complex types of the
same floating-point semantics and works as follows:

* `rfft(real_operand) = truncated_result` where
* `complex_operand... = (real_operand..., 0.0)`.
* `complex_result = fft(complex_operand)`.
* `truncated_result = complex_result[:(rank(complex_result) / 2 + 1)]`.

(When the discrete Fourier transform is computed for real operands, the first
`N/2 + 1` elements of the result unambiguously define the rest of the result,
so the result of `rfft` is truncated to avoid computing redundant elements).

For `fft_type = RFFT`, `result` is defined as the final result of a series of L
computations where `L = size(fft_length)`. For example, for `L = 3`:

* `result1[i0, ..., :] = rfft(operand[i0, ..., :])`.
* `result2[i0, ..., :, iR-1] = fft(result1[i0, ..., :, iR-1])`.
* `result[i0, ..., :, iR-2, iR-1] = fft(result2[i0, ..., :, iR-2, iR-1])`.

Finally, given the function `irfft` which has the same type signature and
computes the inverse of `rfft`:

For `fft_type = IRFFT`, `result` is defined as the inverse of the computations
for `fft_type = RFFT`. For example, for `L = 3`:

* `result1[i0, ..., :, iR-2, iR-1] = ifft(operand[i0, ..., :, iR-2, iR-1])`.
* `result2[i0, ..., :, iR-1] = ifft(result1[i0, ..., :, iR-1])`.
* `result[i0, ..., :] = irfft(result2[i0, ..., :])`.

#### Inputs

| Label | Name         | Type                                         | Constraints            |
|-------|--------------|----------------------------------------------|------------------------|
| (I1)  | `operand`    | tensor of floating-point or complex type     | (C1), (C2), (C4), (C5) |
| (I2)  | `fft_type`   | enum of `FFT`, `IFFT`, `RFFT`, and `IRFFT`   | (C2), (C5)             |
| (I3)  | `fft_length` | 1-dimensional tensor constant of type `si64` | (C1), (C3), (C4)       |

#### Outputs

| Name     | Type                                     | Constraints      |
|----------|------------------------------------------|------------------|
| `result` | tensor of floating-point or complex type | (C2), (C4), (C5) |

#### Constraints

* (C1) `size(fft_length) <= rank(operand)`.
* (C2) The relationship between `operand` and `result` element types varies:
  * If `fft_type = FFT`, `element_type(operand)` and `element_type(result)`
    have the same complex type.
  * If `fft_type = IFFT`, `element_type(operand)` and `element_type(result)`
    have the same complex type.
  * If `fft_type = RFFT`, `element_type(operand)` is a floating-point type and
    `element_type(result)` is a complex type of the same floating-point
    semantics.
  * If `fft_type = IRFFT`, `element_type(operand)` is a complex type and
    `element_type(result)` is a floating-point type of the same floating-point
    semantics.
* (C3) `1 <= size(fft_length) <= 3`.
* (C4) If among `operand` and `result`, there is a tensor `real` of a
floating-point type, then `shape(real)[-size(fft_length):] = fft_length`.
* (C5) `shape(result) = shape(operand)` except for:
  * If `fft_type = RFFT`,
    `dim(result, -1) = dim(operand, -1) = 0 ? 0 : dim(operand, -1) / 2 + 1`.
  * If `fft_type = IRFFT`,
    `dim(operand, -1) = dim(result, -1) = 0 ? 0 : dim(result, -1) / 2 + 1`.

#### Examples

```mlir
// %operand: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
%result = "stablehlo.fft"(%operand) {
  fft_type = #stablehlo<fft_type FFT>,
  fft_length = array<i64: 4>
} : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
// %result: [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
```

### floor

#### Semantics

Performs element-wise floor of `operand` tensor and produces a `result` tensor.
Implements the `roundToIntegralTowardNegative` operation from the IEEE-754
specification. For quantized types, performs
`dequantize_op_quantize(floor, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                         | Constraints |
|-------|-----------|--------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                         | Constraints |
|----------|--------------------------------------------------------------|-------------|
| `result` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [-0.8166, -0.2530, 0.2530, 0.8166, 2.0]
%result = "stablehlo.floor"(%operand) : (tensor<5xf32>) -> tensor<5xf32>
// %result: [-1.0, -1.0, 0.0, 0.0, 2.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/floor.mlir)

### gather

#### Semantics

Gathers slices from `operand` tensor from offsets specified in `start_indices`
and produces a `result` tensor.

The following diagram shows how elements in `result` map on elements in
`operand` using a concrete example. The diagram picks a few example `result`
indices and explains in detail which `operand` indices they correspond to.

![gather](images/spec/gather.svg)

More formally, `result[result_index] = operand[operand_index]` where:

<!-- markdownlint-disable line-length -->
* `batch_dims = [d for d in axes(result) and d not in offset_dims]`.
* `batch_index = result_index[batch_dims...]`.
* `start_index` is defined as:
  * `start_indices[bi0, ..., :, ..., biN]` where `bi` are individual elements in
    `batch_index` and `:` is inserted at the `index_vector_dim` index, if
    `index_vector_dim` < `rank(start_indices)`.
  * `[start_indices[batch_index]]` otherwise.
* For `d_operand` in `axes(operand)`,
  * `full_start_index[d_operand] = clamp(start_index[d_start], 0,
    dim(operand, d_operand) - slice_sizes[d_operand])`
    if `d_operand = start_index_map[d_start]`.
  * `full_start_index[d_operand] = 0` otherwise.
* For `d_operand` in `axes(operand)`,
  * `full_batching_index[d_operand] =
    batch_index[d_start - (d_start < index_vector_dim ? 0 : 1)]`
    if `d_operand = operand_batching_dims[i_batching]` and
    `d_start = start_indices_batching_dims[i_batching]`.
  * `full_batching_index[d_operand] = 0` otherwise.
* `offset_index = result_index[offset_dims...]`.
* `full_offset_index = [oi0, ..., 0, ..., oiN]` where `oi` are individual
  elements in `offset_index`, and `0` is inserted at indices from
  `collapsed_slice_dims` and `operand_batching_dims`.
* `operand_index = full_start_index + full_batching_index + full_offset_index`.
<!-- markdownlint-enable line-length -->

If `indices_are_sorted` is `true` then the implementation can assume that
`start_indices` are sorted with respect to `start_index_map`, otherwise the
behavior is undefined. More formally, for all `i1 < i2` from `indices(result)`,
`full_start_index(i1) <= full_start_index(i2)`.

#### Inputs

| Label | Name                          | Type                                         | Constraints                                |
|-------|-------------------------------|----------------------------------------------|--------------------------------------------|
| (I1)  | `operand`                     | tensor or per-tensor quantized tensor        | (C1), (C8), (C11), (C17), (C19-C21), (C23) |
| (I2)  | `start_indices`               | tensor of integer type                       | (C2-C3), (C14), (C17), (C22)               |
| (I3)  | `offset_dims`                 | 1-dimensional tensor constant of type `si64` | (C1), (C4-C5), (C22)                       |
| (I4)  | `collapsed_slice_dims`        | 1-dimensional tensor constant of type `si64` | (C1), (C6-C9), (C22)                       |
| (I5)  | `operand_batching_dims`       | 1-dimensional tensor constant of type `si64` | (C1), (C6), (C10-C12), (C16-C18), (C22)    |
| (I6)  | `start_indices_batching_dims` | 1-dimensional tensor constant of type `si64` | (C13-C17)                                  |
| (I7)  | `start_index_map`             | 1-dimensional tensor constant of type `si64` | (C3), (C18-C19)                            |
| (I8)  | `index_vector_dim`            | constant of type `si64`                      | (C2-C3), (C15), (C22)                      |
| (I9)  | `slice_sizes`                 | 1-dimensional tensor constant of type `si64` | (C9), (C12), (C20-C22)                     |
| (I10) | `indices_are_sorted`          | constant of type `i1`                        |                                            |

#### Outputs

| Name     | Type                                  | Constraints     |
|----------|---------------------------------------|-----------------|
| `result` | tensor or per-tensor quantized tensor | (C5), (C22-C23) |

#### Constraints

* (C1) `rank(operand) = size(offset_dims) + size(collapsed_slice_dims) +
       size(operand_batching_dims)`.
* (C2) `0 <= index_vector_dim <= rank(start_indices)`.
* (C3) `size(start_index_map) =
       index_vector_dim < rank(start_indices) ?
       dim(start_indices, index_vector_dim) : 1`.
* (C4) `is_unique(offset_dims) and is_sorted(offset_dims)`.
* (C5) `0 <= offset_dims < rank(result)`.
* (C6) `is_unique(concatenate(collapsed_slice_dims, operand_batching_dims))`
* (C7) `is_sorted(collapsed_slice_dims)`.
* (C8) `0 <= collapsed_slice_dims < rank(operand)`.
* (C9) `slice_sizes[collapsed_slice_dims...] <= 1`.
* (C10) `is_sorted(operand_batching_dims)`.
* (C11) `0 <= operand_batching_dims < rank(operand)`.
* (C12) `slice_sizes[operand_batching_dims...] <= 1`.
* (C13) `is_unique(start_indices_batching_dims)`.
* (C14) `0 <= start_indices_batching_dims < rank(start_indices)`.
* (C15) `index_vector_dim not in start_indices_batching_dims`.
* (C16) `size(operand_batching_dims) == size(start_indices_batching_dims)`.
* (C17) `dim(operand, operand_batching_dims...) =
        dim(start_indices, start_indices_batching_dims...)`.
* (C18) `is_unique(concatenate(start_index_map, operand_batching_dims))`.
* (C19) `0 <= start_index_map < rank(operand)`.
* (C20) `size(slice_sizes) = rank(operand)`.
* (C21) `0 <= slice_sizes <= shape(operand)`.
* (C22) `shape(result) = combine(batch_dim_sizes, offset_dim_sizes)` where:
  * `batch_dim_sizes = shape(start_indices)` except that the dimension size
    of `start_indices` corresponding to `index_vector_dim` is not included.
  * `offset_dim_sizes = slice_sizes` except that the dimension sizes in
    `slice_sizes` corresponding to `collapsed_slice_dims` and
    `operand_batching_dims` are not included.
  * `combine` puts `batch_dim_sizes` at axes corresponding to `batch_dims` and
   `offset_dim_sizes` at axes corresponding to `offset_dims`.
* (C23) `element_type(operand) = element_type(result)`.

#### Examples

```mlir
// %operand: [
//            [
//             [[1, 2], [3, 4], [5, 6], [7, 8]],
//             [[9, 10],[11, 12], [13, 14], [15, 16]],
//             [[17, 18], [19, 20], [21, 22], [23, 24]]
//            ],
//            [
//             [[25, 26], [27, 28], [29, 30], [31, 32]],
//             [[33, 34], [35, 36], [37, 38], [39, 40]],
//             [[41, 42], [43, 44], [45, 46], [47, 48]]
//            ]
//           ]
// %start_indices: [
//                  [
//                   [[0, 0], [1, 0], [2, 1]],
//                   [[0, 1], [1, 1], [0, 9]]
//                  ],
//                  [
//                   [[0, 0], [2, 1], [2, 2]],
//                   [[1, 2], [0, 1], [1, 0]]
//                  ]
//                 ]
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
} : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
// %result: [
//           [
//            [
//             [[1, 2], [3, 4]],
//             [[3, 4], [5, 6]],
//             [[13, 14], [15, 16]]
//            ],
//            [
//             [[33, 34], [35, 36]],
//             [[35, 36], [37, 38]],
//             [[41, 42], [43, 44]]
//            ]
//           ],
//           [
//            [
//             [[1, 2], [3, 4]],
//             [[13, 14], [15, 16]],
//             [[21, 22], [23, 24]]
//            ],
//            [
//             [[43, 44], [45, 46]],
//             [[33, 34], [35, 36]],
//             [[27, 28], [29, 30]]
//            ]
//           ]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/gather.mlir)

### get_dimension_size

#### Semantics

Produces the size of the given `dimension` of the `operand`. More formally,
`result = dim(operand, dimension)`. The Semantics concerns only with the shape
component of the type. The element-type could be anything.

#### Inputs

| Label | Name        | Type                       | Constraints |
|-------|-------------|----------------------------|-------------|
| (I1)  | `operand`   | tensor or quantized tensor | (C1)        |
| (I2)  | `dimension` | constant of type `si64`    | (C1)        |

#### Outputs

| Name     | Type                                |
|----------|-------------------------------------|
| `result` | 0-dimensional tensor of type `si32` |

#### Constraints

* (C1) `0 <= dimension < rank(operand)`.

#### Examples

```mlir
// %operand: [[1, 2, 3], [4, 5, 6]]
%result = "stablehlo.get_dimension_size"(%operand) {
  dimension = 1 : i64
} : (tensor<2x3xi64>) -> tensor<i32>
// %result: 3
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/get_dimension_size.mlir)

### get_tuple_element

> Note: Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283),
> this op is being explored for deprecation as it appears to be unused by both
> frameworks and compilers. As such, it has limited compatibility guarantees
> (6 months).

#### Semantics

Extracts element at `index` position of the `operand` tuple and produces a
`result`. More formally, `result = operand[index]`.

#### Inputs

| Label | Name      | Type                    | Constraints |
|-------|-----------|-------------------------|-------------|
| (I1)  | `operand` | tuple                   | (C1), (C2)  |
| (I2)  | `index`   | constant of type `si32` | (C1), (C2)  |

#### Outputs

| Name     | Type               | Constraints |
|----------|--------------------|-------------|
| `result` | any supported type | (C2)        |

#### Constraints

* (C1) `0 <= index < size(operand)`.
* (C2) `type(result) = tuple_element_types(operand)[index]`.

#### Examples

```mlir
// %operand: ([1.0, 2.0], (3))
  index = 0 : i32
} : (tuple<tensor<2xf32>, tuple<tensor<i32>>>) -> tensor<2xf32>
// %result: [1.0, 2.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/tuple_and_get_tuple_element.mlir)

### if

#### Semantics

Produces the output from executing exactly one function from `true_branch` or
`false_branch` depending on the value of `pred`. More formally, `result =
pred ? true_branch() : false_branch()`.

#### Inputs

| Label | Name           | Type                              | Constraints |
|-------|----------------|-----------------------------------|-------------|
| (I1)  | `pred`         | 0-dimensional tensor of type `i1` |             |
| (I2)  | `true_branch`  | function                          | (C1-C3)     |
| (I3)  | `false_branch` | function                          | (C1), (C2)  |

#### Outputs

| Name      | Type                                                    | Constraints |
|-----------|---------------------------------------------------------|-------------|
| `results` | variadic number of tensors, quantized tensors or tokens | (C3)        |

#### Constraints

* (C1) `input_types(true_branch) = input_types(false_branch) = []`.
* (C2) `output_types(true_branch) = output_types(false_branch)`.
* (C3) `type(results...) = output_types(true_branch)`.

#### Examples

```mlir
// %result_true_branch: 10
// %result_false_branch: 11
// %pred: true
%result = "stablehlo.if"(%pred) ({
  "stablehlo.return"(%result_true_branch) : (tensor<i32>) -> ()
}, {
  "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
}) : (tensor<i1>) -> tensor<i32>
// %result: 10
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/if.mlir)

### imag

#### Semantics

Extracts the imaginary part, element-wise, from the `operand` and produces a
`result` tensor. More formally, for each element `x`:
`imag(x) = is_complex(x) ? imaginary_part(x) :
constant(0, element_type(result))`.

#### Inputs

| Label | Name      | Type                                     | Constraints |
|-------|-----------|------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type | (C1), (C2)  |

#### Outputs

| Name     | Type                          | Constraints |
|----------|-------------------------------|-------------|
| `result` | tensor of floating-point type | (C1), (C2)  |

#### Constraints

* (C1) `shape(result) = shape(operand)`.
* (C2) `element_type(result)` is defined as:
  * `complex_element_type(element_type(operand))` if `is_complex(operand)`.
  * `element_type(operand)` otherwise.

#### Examples

```mlir
// %operand: [(1.0, 2.0), (3.0, 4.0)]
%result = "stablehlo.imag"(%operand) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// %result: [2.0, 4.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/imag.mlir)

### infeed

#### Semantics

Reads data from the infeed and produces `results`.

Semantics of `infeed_config` is implementation-defined.

`results` consist of payload values which come first and a token which comes
last. In the future, we are planning to split the payload and the token into two
separate outputs to improve clarity
([#670](https://github.com/openxla/stablehlo/issues/670)).

#### Inputs

| Label | Name            | Type                      |
|-------|-----------------|---------------------------|
| (I1)  | `token`         | `token`                   |
| (I2)  | `infeed_config` | constant of type `string` |

#### Outputs

| Name      | Type                                                    | Constraints |
|-----------|---------------------------------------------------------|-------------|
| `results` | variadic number of tensors, quantized tensors or tokens | (C1-C3)     |

#### Constraints

* (C1) `0 < size(results)`.
* (C2) `is_empty(result[:-1])` or `is_tensor(type(results[:-1]))`.
* (C3) `is_token(type(results[-1]))`.

#### Examples

```mlir
// %token: !stablehlo.token
// infeed_queue[0]: [[1, 2], [3, 4]]
// infeed_queue[1]: [[5, 6], [7, 8]]
%results0:2 = "stablehlo.infeed"(%token) {
  infeed_config = ""
} : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
// results0#0: [[1, 2], [3, 4]]
%results1:2 = "stablehlo.infeed"(%token) {
  infeed_config = ""
} : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
// results1#0: [[5, 6], [7, 8]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/infeed.mlir)

### iota

#### Semantics

Fills an `output` tensor with values in increasing order starting from zero
along the `iota_dimension` dimension. More formally,

`output[output_index] = constant(is_quantized(output) ?
quantize(output_index[iota_dimension], element_type(output)) :
output_index[iota_dimension], element_type(output))`.

#### Inputs

| Label | Name             | Type   | Constraints |
|-------|------------------|--------|-------------|
| (I1)  | `iota_dimension` | `si64` | (C1)        |

#### Outputs

| Name     | Type                                                                              | Constraints |
|----------|-----------------------------------------------------------------------------------|-------------|
| `output` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `0 <= iota_dimension < rank(output)`.

#### Examples

```mlir
%output = "stablehlo.iota"() {
  iota_dimension = 0 : i64
} : () -> tensor<4x5xi32>
// %output: [
//           [0, 0, 0, 0, 0],
//           [1, 1, 1, 1, 1],
//           [2, 2, 2, 2, 2],
//           [3, 3, 3, 3, 3]
//          ]

%output = "stablehlo.iota"() {
  iota_dimension = 1 : i64
} : () -> tensor<4x5xi32>
// %output: [
//           [0, 1, 2, 3, 4],
//           [0, 1, 2, 3, 4],
//           [0, 1, 2, 3, 4],
//           [0, 1, 2, 3, 4]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/iota.mlir)

### is_finite

#### Semantics

Performs element-wise check whether the value in `x` is finite (i.e. is neither
+Inf, -Inf, nor NaN) and produces a `y` tensor. Implements the `isFinite`
operation from the IEEE-754 specification. For quantized types, the result is
always `true`.

#### Inputs

| Label | Name | Type                                                         | Constraints |
|-------|------|--------------------------------------------------------------|-------------|
| (I1)  | `x`  | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name | Type                   | Constraints |
|------|------------------------|-------------|
| `y`  | tensor of boolean type | (C1)        |

#### Constraints

* (C1) `shape(x) = shape(y)`.

#### Examples

```mlir
// Logical values: -Inf, +Inf, NaN, ...
// %x: [0xFFF0000000000000, 0x7FF0000000000000, 0x7FF8000000000000, -10.0, -0.0, 0.0, 10.0]
%y = "stablehlo.is_finite"(%x) : (tensor<7xf64) -> tensor<7xi1>
// %y: [false, false, false, true, true, true, true]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/is_finite.mlir)

### log

#### Semantics

Performs element-wise logarithm operation on `operand` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `log` from IEEE-754.
* For complex numbers: complex logarithm.
* For quantized types: `dequantize_op_quantize(log, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [[1.0, 2.0], [3.0, 4.0]]
%result = "stablehlo.log"(%operand) : (tensor<2x2xf64>) -> tensor<2x2xf64>
// %result: [[0.0, 0.69314718055994529], [1.0986122886681098, 1.3862943611198906]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/log.mlir)

### log_plus_one

#### Semantics

Performs element-wise logarithm plus one operation on `operand` tensor and
produces a `result` tensor. Depending on the element type, does the following:

* For floats: `logp1` from IEEE-754.
* For complex numbers: complex logarithm plus one.
* For quantized types:
  `dequantize_op_quantize(log_plus_one, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [0.0, -0.999, 7.0, 6.38905621, 15.0]
%result = "stablehlo.log_plus_one"(%operand) : (tensor<5xf64>) -> tensor<5xf64>
// %result: [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/log_plus_one.mlir)

### logistic

#### Semantics

Performs element-wise logistic operation on `operand` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `division(1, addition(1, exp(-x)))` from IEEE-754.
* For complex numbers: complex logistic.
* For quantized types:
  `dequantize_op_quantize(logistic, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [[0.0, 1.0], [2.0, 3.0]]
%result = "stablehlo.logistic"(%operand) : (tensor<2x2xf64>) -> tensor<2x2xf64>
// %result: [[0.5, 0.73105858], [0.88079708, 0.95257413]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/logistic.mlir)

### map

> Note: Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283),
> this op is being explored for deprecation as it appears to be unused by both
> frameworks and compilers. As such, it has limited compatibility guarantees
> (6 months).

#### Semantics

Applies a map function `computation` to `inputs` along the `dimensions` and
produces a `result` tensor.

More formally, `result[result_index] = computation(inputs...[result_index])`.

#### Inputs

| Label | Name          | Type                                                       | Constraints |
|-------|---------------|------------------------------------------------------------|-------------|
| (I1)  | `inputs`      | variadic number of tensors or per-tensor quantized tensors | (C1-C4)     |
| (I2)  | `dimensions`  | 1-dimensional tensor constant of type `si64`               | (C3)        |
| (I3)  | `computation` | function                                                   | (C4)        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1), (C4)  |

#### Constraints

* (C1) `shape(inputs...) = shape(result)`.
* (C2) `0 < size(inputs) = N`.
* (C3) `dimensions = range(rank(inputs[0]))`.
* (C4) `computation` has type `(tensor<E0>, ..., tensor<EN-1>) -> tensor<E'>`
  where `Ei = element_type(inputs[i])` and `E' = element_type(result)`.

#### Examples

```mlir
// %input0: [[0, 1], [2, 3]]
// %input1: [[4, 5], [6, 7]]
%result = "stablehlo.map"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
    stablehlo.return %0 : tensor<i64>
}) {
  dimensions = array<i64: 0, 1>
} : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
// %result: [[0, 5], [12, 21]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/map.mlir)

### maximum

#### Semantics

Performs element-wise max operation on tensors `lhs` and `rhs` and produces a
`result` tensor. Depending on the element type, does the following:

* For booleans: logical OR.
* For integers: integer maximum.
* For floats: `maximum` from IEEE-754.
* For complex numbers: lexicographic maximum for the `(real, imaginary)` pair.
  Imposing an ordering on complex numbers involves surprising semantics,
  so in the future we are planning to remove support for complex numbers
  for this operation ([#560](https://github.com/openxla/stablehlo/issues/560)).
* For quantized types:
  * `dequantize_op_quantize(maximum, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                                  | Constraints |
|-------|-------|---------------------------------------|-------------|
| (I1)  | `lhs` | tensor or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(lhs) = baseline_type(rhs) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [[1, 2], [7, 8]]
// %rhs: [[5, 6], [3, 4]]
%result = "stablehlo.maximum"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[5, 6], [7, 8]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/maximum.mlir)

### minimum

#### Semantics

Performs element-wise min operation on tensors `lhs` and `rhs` and produces a
`result` tensor. Depending on the element type, does the following:

* For booleans: logical AND.
* For integers: integer minimum.
* For floats: `minimum` from IEEE-754.
* For complex numbers: lexicographic minimum for the `(real, imaginary)` pair.
  Imposing an ordering on complex numbers involves surprising semantics,
  so in the future we are planning to remove support for complex numbers
  for this operation ([#560](https://github.com/openxla/stablehlo/issues/560)).
* For quantized types:
  * `dequantize_op_quantize(minimum, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                                  | Constraints |
|-------|-------|---------------------------------------|-------------|
| (I1)  | `lhs` | tensor or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(lhs) = baseline_type(rhs) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [[1, 2], [7, 8]]
// %rhs: [[5, 6], [3, 4]]
%result = "stablehlo.minimum"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[1, 2], [3, 4]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/minimum.mlir)

### multiply

#### Semantics

Performs element-wise product of two tensors `lhs` and `rhs` and produces a
`result` tensor. Depending on the element type, does the following:

* For booleans: logical AND.
* For integers: integer multiplication.
* For floats: `multiplication` from IEEE-754.
* For complex numbers: complex multiplication.
* For quantized types:
  * `dequantize_op_quantize(multiply, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                                  | Constraints |
|-------|-------|---------------------------------------|-------------|
| (I1)  | `lhs` | tensor or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [[1, 2], [3, 4]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[5, 12], [21, 32]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/multiply.mlir)

### negate

#### Semantics

Performs element-wise negation of `operand` tensor and produces a `result`
tensor. Depending on the element type, does the following:

* For signed integers: integer negation.
* For unsigned integers: bitcast to signed integer, integer negation, bitcast
  back to unsigned integer.
* For floats: `negate` from IEEE-754.
* For complex numbers: complex negation.
* For quantized types:
  `dequantize_op_quantize(negate, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                              | Constraints |
|-------|-----------|-----------------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                              | Constraints |
|----------|-----------------------------------------------------------------------------------|-------------|
| `result` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

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

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/negate.mlir)

### not

#### Semantics

Performs element-wise NOT of tensor `operand` and produces a `result` tensor.
Depending on the element type, does the following:

* For booleans: logical NOT.
* For integers: bitwise NOT.

#### Arguments

| Name      | Type                              | Constraints |
|-----------|-----------------------------------|-------------|
| `operand` | tensor of boolean or integer type | (C1)        |

#### Outputs

| Name     | Type                              | Constraints |
|----------|-----------------------------------|-------------|
| `result` | tensor of boolean or integer type | (C1)        |

#### Constraints

* (C1) `type(operand) = type(result)`.

#### Examples

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

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/not.mlir)

### optimization_barrier

#### Semantics

Ensures that the operations that produce the `operand` are executed before any
operations that depend on the `result` and prevents compiler transformations
from moving operations across the barrier. Other than that, the operation is
an identity, i.e. `result = operand`.

#### Arguments

| Name      | Type                                                               | Constraints |
|-----------|--------------------------------------------------------------------|-------------|
| `operand` | variadic number of tensors, per-tensor quantized tensors or tokens | (C1)        |

#### Outputs

| Name     | Type                                                               | Constraints |
|----------|--------------------------------------------------------------------|-------------|
| `result` | variadic number of tensors, per-tensor quantized tensors or tokens | (C1)        |

#### Constraints

* (C1) `type(operand...) = type(result...)`.

#### Examples

```mlir
// %operand0: 0.0
// %operand1: 1.0
%result0, %result1 = "stablehlo.optimization_barrier"(%operand0, %operand1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
// %result0: 0.0
// %result1: 1.0
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/optimization_barrier.mlir)

### or

#### Semantics

Performs element-wise OR of two tensors `lhs` and `rhs` and produces a `result`
tensor. Depending on the element type, does the following:

* For booleans: logical OR.
* For integers: bitwise OR.

#### Inputs

| Label | Name  | Type                              | Constraints |
|-------|-------|-----------------------------------|-------------|
| (I1)  | `lhs` | tensor of integer or boolean type | (C1)        |
| (I2)  | `rhs` | tensor of integer or boolean type | (C1)        |

#### Outputs

| Name     | Type                              | Constraints |
|----------|-----------------------------------|-------------|
| `result` | tensor of integer or boolean type | (C1)        |

#### Constraints

* (C1) `type(lhs) = type(rhs) = type(result)`.

#### Examples

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

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/or.mlir)

### outfeed

#### Semantics

Writes `inputs` to the outfeed and produces a `result` token.

Semantics of `outfeed_config` is implementation-defined.

#### Inputs

| Label | Name             | Type                                            |
|-------|------------------|-------------------------------------------------|
| (I1)  | `inputs`         | variadic number of tensors or quantized tensors |
| (I2)  | `token`          | `token`                                         |
| (I3)  | `outfeed_config` | constant of type `string`                       |

#### Outputs

| Name     | Type    |
|----------|---------|
| `result` | `token` |

#### Examples

```mlir
%result = "stablehlo.outfeed"(%input0, %token) {
  outfeed_config = ""
} : (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/outfeed.mlir)

### pad

#### Semantics

Expands `operand` by padding around the tensor as well as between the elements
of the tensor with the given `padding_value`.

`edge_padding_low` and `edge_padding_high` specify the amount of padding added
at the low-end (next to index 0) and the high-end (next to the highest index) of
each dimension respectively. The amount of padding can be negative, where the
absolute value of negative padding indicates the number of elements to remove
from the specified dimension.

`interior_padding` specifies the amount of padding added between any two
elements in each dimension which may not be negative. Interior padding occurs
before edge padding such that negative edge padding will remove elements from
the interior-padded operand.

More formally, `result[result_index]` is defined as:

* `operand[operand_index]` if
  `result_index = edge_padding_low + operand_index * (interior_padding + 1)`.
* `padding_value` otherwise.

#### Inputs

| Label | Name                | Type                                                | Constraints      |
|-------|---------------------|-----------------------------------------------------|------------------|
| (I1)  | `operand`           | tensor or per-tensor quantized tensor               | (C1), (C2), (C4) |
| (I2)  | `padding_value`     | 0-dimensional tensor or per-tensor quantized tensor | (C1)             |
| (I3)  | `edge_padding_low`  | 1-dimensional tensor constant of type `si64`        | (C1), (C4)       |
| (I4)  | `edge_padding_high` | 1-dimensional tensor constant of type `si64`        | (C1), (C4)       |
| (I5)  | `interior_padding`  | 1-dimensional tensor constant of type `si64`        | (C2-C4)          |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C3-C6)     |

#### Constraints

* (C1) `element_type(operand) = element_type(padding_value) =
  element_type(result)`.
* (C2) `size(edge_padding_low) = size(edge_padding_high) =
  size(interior_padding) = rank(operand)`.
* (C3) `0 <= interior_padding`.
* (C4) `shape(result) = shape(operand) + edge_padding_low +
  max(shape(operand) - 1, 0) * interior_padding + edge_padding_high`.

#### Examples

```mlir
// %operand: [
//            [1, 2, 3],
//            [4, 5, 6]
//           ]
// %padding_value: 0
%result = "stablehlo.pad"(%operand, %padding_value) {
  edge_padding_low = array<i64: 0, 1>,
  edge_padding_high = array<i64: 2, 1>,
  interior_padding = array<i64: 1, 2>
} : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
// %result: [
//           [0, 1, 0, 0, 2, 0, 0, 3, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 4, 0, 0, 5, 0, 0, 6, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0],
//           [0, 0, 0, 0, 0, 0, 0, 0, 0]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/pad.mlir)

### partition_id

#### Semantics

Produces `partition_id` of the current process.

#### Outputs

| Name     | Type                                |
|----------|-------------------------------------|
| `result` | 0-dimensional tensor of type `ui32` |

#### Examples

```mlir
%result = "stablehlo.partition_id"() : () -> tensor<ui32>
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/partition_id.mlir)

### popcnt

#### Semantics

Performs element-wise count of the number of bits set in the `operand` tensor
and produces a `result` tensor.

#### Inputs

| Label | Name      | Type                   | Constraints |
|-------|-----------|------------------------|-------------|
| (I1)  | `operand` | tensor of integer type | (C1)        |

#### Outputs

| Name     | Type                   | Constraints |
|----------|------------------------|-------------|
| `result` | tensor of integer type | (C1)        |

#### Constraints

* (C1) `type(operand) = type(result)`.

#### Examples

```mlir
// %operand: [0, 1, 2, 127]
%result = "stablehlo.popcnt"(%operand) : (tensor<4xi64>) -> tensor<4xi64>
// %result: [0, 1, 1, 7]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/popcnt.mlir)

### power

#### Semantics

Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
produces a `result` tensor. Depending on the element type, does the following:

* For integers: integer exponentiation.
* For floats: `pow` from IEEE-754.
* For complex numbers: complex exponentiation.
* For quantized types: `dequantize_op_quantize(power, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                                                                              | Constraints |
|-------|-------|-----------------------------------------------------------------------------------|-------------|
| (I1)  | `lhs` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                              | Constraints |
|----------|-----------------------------------------------------------------------------------|-------------|
| `result` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]
// %rhs: [2.0, 2.0, 1.1, 2.0, -1.0, 10.0]
%result = "stablehlo.power"(%lhs, %rhs) : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
// %result: [4.0, 0.0, -nan, 25.0, 0.333333343, inf]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/power.mlir)

### real

#### Semantics

Extracts the real part, element-wise, from the `operand` and produces a `result`
tensor. More formally, for each element `x`:
`real(x) = is_complex(x) ? real_part(x) : x`.

#### Inputs

| Label | Name      | Type                                     | Constraints |
|-------|-----------|------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type | (C1), (C2)  |

#### Outputs

| Name     | Type                          | Constraints |
|----------|-------------------------------|-------------|
| `result` | tensor of floating-point type | (C1), (C2)  |

#### Constraints

* (C1) `shape(result) = shape(operand)`.
* (C2) `element_type(result)` is defined as:
  * `complex_element_type(element_type(operand))` if `is_complex(operand)`.
  * `element_type(operand)` otherwise.

#### Examples

```mlir
// %operand: [(1.0, 2.0), (3.0, 4.0)]
%result = "stablehlo.real"(%operand) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// %result: [1.0, 3.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/real.mlir)

### recv

#### Semantics

Receives data from a channel with `channel_id` and produces `results`.

If `is_host_transfer` is `true`, then the operation transfers data from the
host. Otherwise, it transfers data from another device. What this means is
implementation-defined. This flag duplicates the information provided in
`channel_type`, so in the future we are planning to only keep one of them
([#666](https://github.com/openxla/stablehlo/issues/666)).

`results` consist of payload values which come first and a token which comes
last. In the future, we are planning to split the payload and the token into two
separate outputs to improve clarity
([#670](https://github.com/openxla/stablehlo/issues/670)).

#### Inputs

| Label | Name               | Type                                            | Constraints |
|-------|--------------------|-------------------------------------------------|-------------|
| (I1)  | `token`            | `token`                                         | (C4)        |
| (I2)  | `channel_id`       | constant of type `si64`                         |             |
| (I3)  | `channel_type`     | enum of `DEVICE_TO_DEVICE` and `HOST_TO_DEVICE` | (C1)        |
| (I4)  | `is_host_transfer` | constant of type `i1`                           | (C1)        |

#### Outputs

| Name      | Type                                                    | Constraints |
|-----------|---------------------------------------------------------|-------------|
| `results` | variadic number of tensors, quantized tensors or tokens | (C2-C4)     |

#### Constraints

* (C1) `channel_type` is defined as:
  * `HOST_TO_DEVICE` if `is_host_transfer = true`,
  * `DEVICE_TO_DEVICE` otherwise.
* (C2) `0 < size(results)`.
* (C3) `is_empty(result[:-1])` or `is_tensor(type(results[:-1]))`.
* (C4) `is_token(type(results[-1]))`.

#### Examples

```mlir
%results0, %results1 = "stablehlo.recv"(%token) {
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
  is_host_transfer = true
} : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/send_recv.mlir)

### reduce

#### Semantics

Applies a reduction function `body` to `inputs` and `init_values` along the
`dimensions` and produces `results` tensors.

The order of reductions is implementation-defined, which means that `body` and
`init_values` must form a monoid to guarantee that the operation produces the
same results for all inputs on all implementations. However, this condition
doesn't hold for many popular reductions. E.g. floating-point addition for
`body` and zero for `init_values` don't actually form a monoid because
floating-point addition is not associative.

More formally, `results...[j0, ..., jR-1] = reduce(input_slices_converted)` where:

* `input_slices = inputs...[j0, ..., :, ..., jR-1]`, where `:` are inserted
  at `dimensions`.
* `input_slices_converted = to_destination_type(input_slices...,
  type(func_inputs(body)[:len(func_inputs(body))//2])...)`.
* `init_values_converted = to_destination_type(init_values...,
  type(func_inputs(body)[len(func_inputs(body))//2:])...)`.
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
    `init_values_converted` at implementation-defined positions.

#### Inputs

| Label | Name          | Type                                                                     | Constraints         |
|-------|---------------|--------------------------------------------------------------------------|---------------------|
| (I1)  | `inputs`      | variadic number of tensors or per-tensor quantized tensors               | (C1-C4), (C6), (C7) |
| (I2)  | `init_values` | variadic number of 0-dimensional tensors or per-tensor quantized tensors | (C2), (C3)          |
| (I3)  | `dimensions`  | 1-dimensional tensor constant of type `si64`                             | (C4), (C5), (C7)    |
| (I4)  | `body`        | function                                                                 | (C6)                |

#### Outputs

| Name      | Type                                                       | Constraints      |
|-----------|------------------------------------------------------------|------------------|
| `results` | variadic number of tensors or per-tensor quantized tensors | (C3), (C7), (C8) |

#### Constraints

* (C1) `same(shape(inputs...))`.
* (C2) `element_type(inputs...) = element_type(init_values...)`.
* (C3) `0 < size(inputs) = size(init_values) = size(results) = N`.
* (C4) `0 <= dimensions < rank(inputs[0])`.
* (C5) `is_unique(dimensions)`.
* (C6) `body` has type `(tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ...,`
  `tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>)` where
  `is_promotable(element_type(inputs[i]), Ei)`.
* (C7) `shape(results...) = shape(inputs...)` except that the dimension
  sizes of `inputs...` corresponding to `dimensions` are not included.
* (C8) `element_type(results[i]) = Ei` for all `i` in `[0,N)`.

#### Examples

```mlir
// %input = [[0, 1, 2, 3, 4, 5]]
// %init_value = 0
%result = "stablehlo.reduce"(%input, %init_value) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  dimensions = array<i64: 1>
} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
// %result = [15]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/reduce.mlir)

### reduce_precision

#### Semantics

Performs element-wise conversion of `operand` to another floating-point type
that uses `exponent_bits` and `mantissa_bits` and back to the original
floating-point type and produces an `output` tensor.

More formally:

* The mantissa bits of the original value are updated to round the original
  value to the nearest value representable with `mantissa_bits` using
  `roundToIntegralTiesToEven` semantics.
* Then, if `mantissa_bits` are smaller than the number of mantissa bits of
  the original value, the mantissa bits are truncated to `mantissa_bits`.
* Then, if the exponent bits of the intermediate result don't fit into the
  range provided by `exponent_bits`, the intermediate result overflows to
  infinity using the original sign or underflows to zero using the
  original sign.
* For quantized types, performs `dequantize_op_quantize(
    lambda operand: reduce_precision(operand, exponent_bits, mantissa_bits),
    operand, type(result))`.

#### Inputs

| Label | Name            | Type                                                         | Constraints |
|-------|-----------------|--------------------------------------------------------------|-------------|
| (I1)  | `operand`       | tensor of floating-point type or per-tensor quantized tensor | (C1)        |
| (I2)  | `exponent_bits` | constant of type `si32`                                      | (C2)        |
| (I3)  | `mantissa_bits` | constant of type `si32`                                      | (C3)        |

#### Outputs

| Name     | Type                                                         | Constraints |
|----------|--------------------------------------------------------------|-------------|
| `output` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(output)`.
* (C2) `1 <= exponent_bits`.
* (C3) `0 <= mantissa_bits`.

#### Examples

```mlir
// Logical values: +Inf, NaN, +Denormal, 0.0, 65519.0, 65520.0
// %operand: [0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0.0, 65519.0, 65520.0]
%output = "stablehlo.reduce_precision"(%operand) {
  exponent_bits = 5 : i32,
  mantissa_bits = 10 : i32
} : (tensor<6xf64>) -> tensor<6xf64>
// Logical values: +Inf, NaN, 0.0, 0.0, 65504.0, +Inf
// %output: [0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF, 0.0, 0.0, 65504.0, 0x7FF0000000000000]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/reduce_precision.mlir)

### reduce_scatter

#### Semantics

![reduce_scatter](images/spec/reduce_scatter.svg)

Within each process group in the StableHLO process grid, performs reduction,
using `computations`, over the values of the `operand` tensor from each process,
splits the reduction result along `scatter_dimension` into parts, and scatters
the split parts between the processes to produce the `result`.

The operation splits the StableHLO process grid into `process_groups` which is
defined as follows:

* `cross_replica(replica_groups)`
  if `channel_id <= 0 and use_global_device_ids = false`.
* `cross_replica_and_partition(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = false`.
* `flattened_ids(replica_groups)`
  if `channel_id > 0 and use_global_device_ids = true`.

Afterwards, within each `process_group`:

* `reduced_value = all_reduce(operand, replica_groups, channel_id,
  use_global_device_ids, computation)`.
* `parts@sender = split(reduced_value@sender, dim(process_groups, 1),
  scatter_dimension)`.
* `result@receiver = parts@sender[receiver_index]` for all `sender` in
  `process_group`, where `receiver_index = process_group.index(receiver)`.

#### Inputs

| Label | Name                    | Type                                         | Constraints            |
|-------|-------------------------|----------------------------------------------|------------------------|
| (I1)  | `operand`               | tensor or per-tensor quantized tensor        | (C1), (C2), (C7), (C8) |
| (I2)  | `scatter_dimension`     | constant of type `si64`                      | (C1), (C2), (C8)       |
| (I3)  | `replica_groups`        | 2-dimensional tensor constant of type `si64` | (C3-C5)                |
| (I4)  | `channel_id`            | constant of type `si64`                      | (C6)                   |
| (I5)  | `use_global_device_ids` | constant of type `i1`                        | (C6)                   |
| (I6)  | `computation`           | function                                     | (C7)                   |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C8-C9)     |

#### Constraints

* (C1) `dim(operand, scatter_dimension) % dim(process_groups, 1) = 0`.
* (C2) `0 <= scatter_dimension < rank(operand)`.
* (C3) `is_unique(replica_groups)`.
* (C4) `size(replica_groups)` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_replicas` if `cross_replica_and_partition` is used.
  * `num_processes` if `flattened_ids` is used.
* (C5) `0 <= replica_groups < size(replica_groups)`.
* (C6) If `use_global_device_ids = true`, then `channel_id > 0`.
* (C7) `computation` has type `(tensor<E>, tensor<E>) -> (tensor<E>)` where
       `is_promotable(element_type(operand), E)`.
* (C8) `shape(result) = shape(operand)` except:
  * `dim(result, scatter_dimension) = dim(operand, scatter_dimension) /
    dim(process_groups, 1)`.
* (C9) `element_type(result) = E`.

#### Examples

```mlir
// num_replicas: 2
// num_partitions: 1
// %operand@(0, 0): [[1, 2, 3, 4],
//                   [5, 6, 7, 8]]
// %operand@(1, 0): [[9, 10, 11, 12],
//                   [13, 14, 15, 16]]
%result = "stablehlo.reduce_scatter"(%operand) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  scatter_dimension = 1 : i64,
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x4xi64>) -> tensor<2x2xi64>
//
// %result@(0, 0): [[10, 12],
//                  [18, 20]]
// %result@(1, 0): [[14, 16],
//                  [22, 24]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/reduce_scatter.mlir)

### reduce_window

#### Semantics

Applies a reduction function `body` to windows of `inputs` and `init_values`
and produces `results`.

The following diagram shows how elements in `results...` are computed from
`inputs...` using a concrete example.

![reduce_window](images/spec/reduce_window.svg)

More formally,
`results...[result_index] = reduce(windows, init_values, axes(inputs...), body)`
(see [reduce](#reduce)) where:

* `padded_inputs = pad(inputs..., init_values..., padding[:, 0], padding[:, 1],
  base_dilations - 1)`.
* `window_start = result_index * window_strides`.
* `window_end = window_start + (window_dimensions - 1) * window_dilations + 1`.
* `windows = slice(padded_inputs..., window_start, window_end,
  window_dilations)`.

#### Inputs

| Label | Name                | Type                                                                     | Constraints                                     |
|-------|---------------------|--------------------------------------------------------------------------|-------------------------------------------------|
| (I1)  | `inputs`            | variadic number of tensors or per-tensor quantized tensors               | (C1-C4), (C6), (C8), (C10), (C12), (C13), (C15) |
| (I2)  | `init_values`       | variadic number of 0-dimensional tensors or per-tensor quantized tensors | (C1), (C13)                                     |
| (I3)  | `window_dimensions` | 1-dimensional tensor constant of type `si64`                             | (C4), (C5), (C15)                               |
| (I4)  | `window_strides`    | 1-dimensional tensor constant of type `si64`                             | (C6), (C7), (C15)                               |
| (I5)  | `base_dilations`    | 1-dimensional tensor constant of type `si64`                             | (C8), (C9), (C15)                               |
| (I6)  | `window_dilations`  | 1-dimensional tensor constant of type `si64`                             | (C10), (C11), (C15)                             |
| (I7)  | `padding`           | 2-dimensional tensor constant of type `si64`                             | (C12), (C15)                                    |
| (I8)  | `body`              | function                                                                 | (C13)                                           |

#### Outputs

| Name      | Type                                                       | Constraints     |
|-----------|------------------------------------------------------------|-----------------|
| `results` | variadic number of tensors or per-tensor quantized tensors | (C1), (C14-C16) |

#### Constraints

<!-- markdownlint-disable line-length -->
* (C1) `0 < size(inputs) = size(init_values) = size(results) = N`.
* (C2) `same(shape(inputs...))`.
* (C3) `element_type(inputs...) = element_type(init_values...)`.
* (C4) `size(window_dimensions) = rank(inputs[0])`.
* (C5) `0 < window_dimensions`.
* (C6) `size(window_strides) = rank(inputs[0])`.
* (C7) `0 < window_strides`.
* (C8) `size(base_dilations) = rank(inputs[0])`.
* (C9) `0 < base_dilations`.
* (C10) `size(window_dilations) = rank(inputs[0])`.
* (C11) `0 < window_dilations`.
* (C12) `shape(padding) = [rank(inputs[0]), 2]`.
* (C13) `body` has type `(tensor<E0>, ..., tensor<EN-1>, tensor<E0>, ...,`
  `tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>)` where
  `is_promotable(element_type(inputs[i]), Ei)`.
* (C14) `same(shape(results...))`.
* (C15) `shape(results[0]) = num_windows` where:
  * `dilated_input_shape = shape(inputs[0]) = 0 ? 0 : (shape(inputs[0]) - 1) * base_dilations + 1`.
  * `padded_input_shape = padding[:, 0] + dilated_input_shape + padding[:, 1]`.
  * `dilated_window_shape = (window_dimensions - 1) * window_dilations + 1`.
  * `is_empty_window = padded_input_shape = 0 || dilated_window_shape > padded_input_shape`.
  * `num_windows = is_empty_window ? 0 : floor((padded_input_shape - dilated_window_shape) / window_strides) + 1`.
* (C16) `element_type(results[i]) = Ei` for all `i` in `[0,N)`.
<!-- markdownlint-enable line-length -->

#### Examples

```mlir
// %input = [[1, 2], [3, 4], [5, 6]]
// %init_value = 0
%result = "stablehlo.reduce_window"(%input, %init_value) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  window_dimensions = array<i64: 2, 1>,
  window_strides = array<i64: 4, 1>,
  base_dilations = array<i64: 2, 1>,
  window_dilations = array<i64: 3, 1>,
  padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
} : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
// %result = [[0, 0], [3, 4]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/reduce_window.mlir)

### remainder

#### Semantics

Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors and
produces a `result` tensor.

More formally, the sign of the result is taken from the dividend, and the
absolute value of the result is always less than the divisor's absolute value.
The remainder is calculated as `lhs - d * rhs`, where `d` is given by:

* For integers: `stablehlo.divide(lhs, rhs)`.
* For floats: `division(lhs, rhs)` from IEEE-754 with rounding attribute
  `roundTowardZero`.
* For complex numbers: TBD
  ([#997](https://github.com/openxla/stablehlo/issues/997)).
* For quantized types:
  * `dequantize_op_quantize(remainder, lhs, rhs, type(result))`.

For floating-point element types, this operation is in contrast with the
`remainder` operation from IEEE-754 specification where `d` is an integral value
nearest to the exact value of `lhs/rhs` with ties to even.

#### Inputs

| Label | Name  | Type                                                                             | Constraints |
|-------|-------|----------------------------------------------------------------------------------|-------------|
| (I1)  | `lhs` | tensor of integer, floating-point or complex type or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor of integer, floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                             | Constraints |
|----------|----------------------------------------------------------------------------------|-------------|
| `result` | tensor of integer, floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [17, -17, 17, -17]
// %rhs: [3, 3, -3, -3]
%result = "stablehlo.remainder"(%lhs, %rhs) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
// %result: [2, -2, 2, -2]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/remainder.mlir)

### replica_id

#### Semantics

Produces `replica_id` of the current process.

#### Outputs

| Name     | Type                                |
|----------|-------------------------------------|
| `result` | 0-dimensional tensor of type `ui32` |

#### Examples

```mlir
%result = "stablehlo.replica_id"() : () -> tensor<ui32>
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/replica_id.mlir)

### reshape

#### Semantics

Performs reshape of `operand` tensor to a `result` tensor. Conceptually, it
amounts to keeping the same canonical representation but potentially changing
the shape, e.g. from `tensor<2x3xf32>` to `tensor<3x2xf32>` or `tensor<6xf32>`.

More formally, `result[result_index] = operand[operand_index]` where
`result_index` and `operand_index` have the same position in the lexicographic
ordering of `index_space(result)` and `index_space(operand)`.

#### Inputs

| Label | Name      | Type                       | Constraints |
|-------|-----------|----------------------------|-------------|
| (I1)  | `operand` | tensor or quantized tensor | (C1-C3)     |

#### Outputs

| Name     | Type                       | Constraints |
|----------|----------------------------|-------------|
| `result` | tensor or quantized tensor | (C1-C3)     |

#### Constraints

* (C1) `element_type(result)` is given by:
  * `element_type(operand)`, if `!is_per_axis_quantized(operand)`.
  * `element_type(operand)` except that `quantization_dimension(operand)` and
    `quantization_dimension(result)` may differ, otherwise.
* (C2) `size(operand) = size(result)`.
* (C3) If `is_per_axis_quantized(operand)`:
  * `reduce(dims(operand, [0, 1, ..., quantization_dimension(operand) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y) =
    reduce(dims(result, [0, 1, ..., quantization_dimension(result) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y)`.
  * `dim(operand, quantization_dimension(operand)) =
    dim(result, quantization_dimension(result))`.
  * `reduce(dims(operand,
    [quantization_dimension(operand) + 1, ..., rank(operand) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y) =
    reduce(dims(result,
    [quantization_dimension(result) + 1, ..., rank(result) - 1]),
    init_values=1, dimensions=[0], body=lambda x, y: x * y)`.

#### Examples

```mlir
// %operand: [[1, 2, 3], [4, 5, 6]]
%result = "stablehlo.reshape"(%operand) : (tensor<2x3xi32>) -> tensor<3x2xi32>
// %result: [[1, 2], [3, 4], [5, 6]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/reshape.mlir)

### reverse

#### Semantics

Reverses the order of elements in the `operand` along the specified `dimensions`
and produces a `result` tensor. More formally,
`result[result_index] = operand[operand_index]` where:

* `operand_index[d] = dim(result, d) - result_index[d] - 1`
  if `d` in `dimensions`.
* `operand_index[d] = result_index[d]` otherwise.

#### Inputs

| Label | Name         | Type                                         | Constraints |
|-------|--------------|----------------------------------------------|-------------|
| (I1)  | `operand`    | tensor or per-tensor quantized tensor        | (C1), (C3)  |
| (I2)  | `dimensions` | 1-dimensional tensor constant of type `si64` | (C2), (C3)  |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1), (C3)  |

#### Constraints

* (C1) `type(operand) = type(result)`.
* (C2) `is_unique(dimensions)`.
* (C3) `0 <= dimensions < rank(result)`.

#### Examples

```mlir
// %operand = [[1, 2], [3, 4], [5, 6]]
%result = "stablehlo.reverse"(%operand) {
  dimensions = array<i64: 1>
} : (tensor<3x2xi32>) -> tensor<3x2xi32>
// %result: [[2, 1], [4, 3], [6, 5]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/reverse.mlir)

### rng

> Note: Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283),
> this op is being explored for deprecation as it appears to be unused by both
> frameworks and compilers. As such, it has limited compatibility guarantees
> (6 months).

#### Semantics

Generates random numbers using the `rng_distribution` algorithm and produces a
`result` tensor of a given shape `shape`.

If `rng_distribution = UNIFORM`, then the random numbers are generated
following the uniform distribution over the interval `[a, b)`. If `a >= b`,
the behavior is undefined.

If `rng_distribution = NORMAL`, then the random numbers are generated
following the normal distribution with mean = `a` and standard deviation = `b`.
If `b < 0`, the behavior is undefined.

The exact way how random numbers are generated is implementation-defined. For
example, they may or may not be deterministic, and they may or may not use
hidden state.

In conversations with many stakeholders, this op has come up as effectively
deprecated, so in the future we are planning to explore removing it
([#597](https://github.com/openxla/stablehlo/issues/597)).

#### Inputs

| Label | Name               | Type                                                             | Constraints |
|-------|--------------------|------------------------------------------------------------------|-------------|
| (I1)  | `a`                | 0-dimensional tensor of integer, boolean, or floating-point type | (C1), (C2)  |
| (I2)  | `b`                | 0-dimensional tensor of integer, boolean, or floating-point type | (C1), (C2)  |
| (I3)  | `shape`            | 1-dimensional tensor constant of type `si64`                     | (C3)        |
| (I4)  | `rng_distribution` | enum of `UNIFORM` and `NORMAL`                                   | (C2)        |

#### Outputs

| Name     | Type                                               | Constraints |
|----------|----------------------------------------------------|-------------|
| `result` | tensor of integer, boolean, or floating-point type | (C1-C3)     |

#### Constraints

* (C1) `element_type(a) = element_type(b) = element_type(result)`.
* (C2) If `rng_distribution = NORMAL`, then `is_float(a)`.
* (C3) `shape(result) = shape`.

#### Examples

```mlir
// %a = 0
// %b = 2
// %shape = [3, 3]
%result = "stablehlo.rng"(%a, %b, %shape) {
  rng_distribution = #stablehlo<rng_distribution UNIFORM>
} : (tensor<i32>, tensor<i32>, tensor<2xi64>) -> tensor<3x3xi32>
// %result: [
//           [1, 0, 1],
//           [1, 1, 1],
//           [0, 0, 0]
//          ]
```

### rng_bit_generator

#### Semantics

Returns an `output` filled with uniform random bits and an updated output state
`output_state` using the pseudorandom number generator algorithm `rng_algorithm`
given an initial state `initial_state`. The output is guaranteed to be
deterministic function of `initial_state`, but it is not guaranteed to be
deterministic between implementations.

`rng_algorithm` is one of the following:

* `DEFAULT`: Implementation-defined algorithm.
* `THREE_FRY`: Implementation-defined variant of the Threefry algorithm.*
* `PHILOX`: Implementation-defined variant of the Philox algorithm.*

\* See: [Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf)

#### Inputs

| Label | Name            | Type                                         | Constraints |
|-------|-----------------|----------------------------------------------|-------------|
| (I1)  | `rng_algorithm` | enum of `DEFAULT`, `THREE_FRY`, and `PHILOX` | (C2)        |
| (I2)  | `initial_state` | 1-dimensional tensor of type `ui64`          | (C1), (C2)  |

#### Outputs

| Name           | Type                                     | Constraints |
|----------------|------------------------------------------|-------------|
| `output_state` | 1-dimensional tensor of type `ui64`      | (C1)        |
| `output`       | tensor of integer or floating-point type |             |

#### Constraints

* (C1) `type(initial_state) = type(output_state)`.
* (C2) `size(initial_state)` is defined as:
  * implementation-defined if `rng_algorithm = DEFAULT`.
  * `2` if `rng_algorithm = THREE_FRY`.
  * `2` or `3` if `rng_algorithm = PHILOX`.

#### Examples

```mlir
// %initial_state: [1, 2]
%output_state, %output = "stablehlo.rng_bit_generator"(%initial_state) {
  rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
// %output_state: [1, 6]
// %output: [
//           [9236835810183407956, 16087790271692313299],
//           [18212823393184779219, 2658481902456610144]
//          ]
```

### round_nearest_afz

#### Semantics

Performs element-wise rounding towards the nearest integer, breaking ties away
from zero, on the `operand` tensor and produces a `result` tensor. Implements
the `roundToIntegralTiesToAway` operation from the IEEE-754 specification. For
quantized types, performs
`dequantize_op_quantize(round_nearest_afz, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                         | Constraints |
|-------|-----------|--------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                         | Constraints |
|----------|--------------------------------------------------------------|-------------|
| `result` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand = [-2.5, 0.4, 0.5, 0.6, 2.5]
%result = "stablehlo.round_nearest_afz"(%operand) : (tensor<5xf64>) -> tensor<5xf64>
// %result: [-3.0, 0.0, 1.0, 1.0, 3.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/round_nearest_afz.mlir)

### round_nearest_even

#### Semantics

Performs element-wise rounding towards the nearest integer, breaking ties
towards the even integer, on the `operand` tensor and produces a `result`
tensor. Implements the `roundToIntegralTiesToEven` operation from the IEEE-754
specification. For quantized types, performs
`dequantize_op_quantize(round_nearest_even, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                         | Constraints |
|-------|-----------|--------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                         | Constraints |
|----------|--------------------------------------------------------------|-------------|
| `result` | tensor of floating-point type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand = [-2.5, 0.4, 0.5, 0.6, 2.5]
%result = "stablehlo.round_nearest_even"(%operand) : (tensor<5xf64>) -> tensor<5xf64>
// %result: [-2.0, 0.0, 0.0, 1.0, 2.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/round_nearest_even.mlir)

### rsqrt

#### Semantics

Performs element-wise reciprocal square root operation on `operand` tensor and
produces a `result` tensor. Depending on the element type, does the following:

* For floats: `rSqrt` from IEEE-754.
* For complex numbers: complex reciprocal square root.
* For quantized types: `dequantize_op_quantize(rsqrt, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [[1.0, 4.0], [9.0, 25.0]]
%result = "stablehlo.rsqrt"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[1.0, 0.5], [0.33333343, 0.2]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/rsqrt.mlir)

### scatter

#### Semantics

Produces `results` tensors which are equal to `inputs` tensors except that
several slices specified by `scatter_indices` are updated with the values
`updates` using `update_computation`.

The following diagram shows how elements in `updates...` map on elements in
`results...` using a concrete example. The diagram picks a few example
`updates...` indices and explains in detail which `results...` indices they
correspond to.

![scatter](images/spec/scatter.svg)

More formally, for all `update_index` in `index_space(updates[0])`:

* `update_scatter_dims = [d for d in axes(updates[0]) and d not in
  update_window_dims]`.
* `update_scatter_index = update_index[update_scatter_dims...]`.
* `start_index` is defined as:
  * `scatter_indices[si0, ..., :, ..., siN]` where `si` are individual
      elements in `update_scatter_index` and `:` is inserted at the
      `index_vector_dim` index, if `index_vector_dim` <
      `rank(scatter_indices)`.
  * `[scatter_indices[update_scatter_index]]` otherwise.
* For `d_input` in `axes(inputs[0])`,
  * `full_start_index[d_input] = start_index[d_start]` if
    `d_input = scatter_dims_to_operand_dims[d_start]`.
  * `full_start_index[d_input] = 0` otherwise.
* For `d_input` in `axes(inputs[0])`,
  * `full_batching_index[d_input] =
    update_scatter_index[d_start - (d_start < index_vector_dim ? 0 : 1)]`
    if `d_input = input_batching_dims[i_batching]` and
    `d_start = scatter_indices_batching_dims[i_batching]`.
  * `full_batching_index[d_input] = 0` otherwise.
* `update_window_index = update_index[update_window_dims...]`.
* `full_window_index = [wi0, ..., 0, ..., wiN]` where `wi` are individual
  elements in `update_window_index`, and `0` is inserted at indices from
  `inserted_window_dims` and `input_batching_dims`.
* `result_index = full_start_index + full_batching_index + full_window_index`.

Given that, `results = exec(schedule, inputs)`, where:

* `schedule` is an implementation-defined permutation of
  `index_space(updates[0])`.
* `exec([update_index, ...], results) = exec([...], updated_results)` where:
  * If `result_index` is in bounds for `shape(results...)`
    * `updates_converted = to_destination_type(
      updates...[update_index], type(func_inputs(update_computation)
      [len(func_inputs(update_computation))//2:])... )`
    * `updated_values = update_computation(results...[result_index],
      updates_converted)`
    * `updated_results` is a copy of `results` with `results...[result_index]`
      set to `updated_values...`.
  * Otherwise
    * `updated_results = results`.
* `exec([], results) = results`.

If `indices_are_sorted` is `true` then the implementation can assume that
`scatter_indices` are sorted with respect to `scatter_dims_to_operand_dims`,
otherwise the behavior is undefined. More formally, for all `i1 < i2` from
`indices(result)`, `full_start_index(i1)` <= `full_start_index(i2)`.

If `unique_indices` is `true` then the implementation can assume that all
`result_index` indices being scattered to are unique. If `unique_indices` is
`true` but the indices being scattered to are not unique then the behavior is
undefined.

#### Inputs

| Label | Name                                  | Type                                                       | Constraints                                                |
|-------|---------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| (I1)  | `inputs`                              | variadic number of tensors or per-tensor quantized tensors | (C1), (C2), (C4-C6), (C11), (C13), (C18), (C21), (C23-C24) |
| (I2)  | `scatter_indices`                     | tensor of integer type                                     | (C4), (C15), (C19), (C22)                                  |
| (I3)  | `updates`                             | variadic number of tensors or per-tensor quantized tensors | (C3-C6), (C8)                                              |
| (I4)  | `update_window_dims`                  | 1-dimensional tensor constant of type `si64`               | (C2), (C4), (C7-C8)                                        |
| (I5)  | `inserted_window_dims`                | 1-dimensional tensor constant of type `si64`               | (C2), (C4), (C9-C11)                                       |
| (I6)  | `input_batching_dims`                 | 1-dimensional tensor constant of type `si64`               | (C2), (C4), (C9), (C12-13), (C17-18), (C20)                |
| (I7)  | `scatter_indices_batching_dims`       | 1-dimensional tensor constant of type `si64`               | (C14-C18)                                                  |
| (I8)  | `scatter_dims_to_operand_dims`        | 1-dimensional tensor constant of type `si64`               | (C19-C21)                                                  |
| (I9)  | `index_vector_dim`                    | constant of type `si64`                                    | (C4), (C16), (C19), (C22)                                  |
| (I10) | `indices_are_sorted`                  | constant of type `i1`                                      |                                                            |
| (I11) | `unique_indices`                      | constant of type `i1`                                      |                                                            |
| (I12) | `update_computation`                  | function                                                   | (C23)                                                      |

#### Outputs

| Name      | Type                                                       | Constraints |
|-----------|------------------------------------------------------------|-------------|
| `results` | variadic number of tensors or per-tensor quantized tensors | (C24-C25)   |

#### Constraints

* (C1) `same(shape(inputs...))`.
* (C2) `rank(inputs[0]) = size(update_window_dims) + size(inserted_window_dims)
       + size(input_batching_dims)`.
* (C3) `same(shape(updates...))`.
* (C4) `shape(updates[0]) = combine(update_scatter_dim_sizes,
       update_window_dim_sizes)` where:
  * `update_scatter_dim_sizes = shape(scatter_indices)` except that
    the dimension size of `scatter_indices` corresponding to
    `index_vector_dim` is not included.
  * `update_window_dim_sizes <= shape(inputs[0])` except that
    the dimension sizes in `inputs[0]` corresponding to `inserted_window_dims`
    and `input_batching_dims` are not included.
  * `combine` puts `update_scatter_dim_sizes` at axes corresponding to
   `update_scatter_dims` and `update_window_dim_sizes` at axes corresponding
   to `update_window_dims`.
* (C5) `0 < size(inputs) = size(updates) = N`.
* (C6) `element_type(updates...) = element_type(inputs...)`.
* (C7) `is_unique(update_window_dims) and is_sorted(update_window_dims)`.
* (C8) `0 <= update_window_dims < rank(updates[0])`.
* (C9) `is_unique(concatenate(inserted_window_dims, input_batching_dims))`
* (C10) `is_sorted(inserted_window_dims)`.
* (C11) `0 <= inserted_window_dims < rank(inputs[0])`.
* (C12) `is_sorted(input_batching_dims)`.
* (C13) `0 <= input_batching_dims < rank(inputs[0]))`.
* (C14) `is_unique(scatter_indices_batching_dims)`.
* (C15) `0 <= scatter_indices_batching_dims < rank(scatter_indices)`.
* (C16) `index_vector_dim not in scatter_indices_batching_dims`.
* (C17) `size(input_batching_dims) == size(scatter_indices_batching_dims)`.
* (C18) `dim(inputs[0], input_batching_dims...) =
        dim(scatter_indices, scatter_indices_batching_dims...)`.
* (C19) `size(scatter_dims_to_operand_dims) =
        index_vector_dim < rank(scatter_indices) ?
        dim(scatter_indices, index_vector_dim) : 1`.
* (C20) `is_unique(concatenate(scatter_dims_to_operand_dims,
        input_batching_dims))`.
* (C21) `0 <= scatter_dims_to_operand_dims < rank(inputs[0])`.
* (C22) `0 <= index_vector_dim <= rank(scatter_indices)`.
* (C23) `update_computation` has type `(tensor<E0>, ..., tensor<EN-1>,
  tensor<E0>, ..., tensor<EN-1>) -> (tensor<E0>, ..., tensor<EN-1>)`,
  where `is_promotable(element_type(inputs[i]), Ei)`.
* (C24) `shape(inputs...) = shape(results...)`.
* (C25) `element_type(results[i]) = Ei` for all `i` in `[0,N)`.

#### Examples

```mlir
// %input: [
//          [
//           [[1, 2], [3, 4], [5, 6], [7, 8]],
//           [[9, 10],[11, 12], [13, 14], [15, 16]],
//           [[17, 18], [19, 20], [21, 22], [23, 24]]
//          ],
//          [
//           [[25, 26], [27, 28], [29, 30], [31, 32]],
//           [[33, 34], [35, 36], [37, 38], [39, 40]],
//           [[41, 42], [43, 44], [45, 46], [47, 48]]
//          ]
//         ]
// %scatter_indices: [
//                    [
//                     [[0, 0], [1, 0], [2, 1]],
//                     [[0, 1], [1, 1], [0, 9]]
//                    ],
//                    [
//                     [[0, 0], [2, 1], [2, 2]],
//                     [[1, 2], [0, 1], [1, 0]]
//                    ]
//                   ]
// %update: [
//           [
//            [[1, 1], [1, 1], [1, 1]],
//            [[1, 1], [1, 1], [1, 1]]
//           ],
//           [
//            [[1, 1], [1, 1], [1, 1]],
//            [[1, 1], [1, 1], [1, 1]]
//           ]
//          ]
%result = "stablehlo.scatter"(%input, %scatter_indices, %update) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
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
// %result: [
//           [
//            [[3, 4], [6, 7], [6, 7], [7, 8]],
//            [[9, 10],[11, 12], [15, 16], [17, 18]],
//            [[17, 18], [19, 20], [22, 23], [24, 25]]
//           ],
//           [
//            [[25, 26], [28, 29], [30, 31], [31, 32]],
//            [[35, 36], [38, 39], [38, 39], [39, 40]],
//            [[41, 42], [44, 45], [46, 47], [47, 48]]
//           ]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/scatter.mlir)

### select

#### Semantics

Produces a `result` tensor where each element is selected from `on_true` or
`on_false` tensor based on the value of the corresponding element of `pred`.
More formally, `result[result_index] = pred_element ? on_true[result_index] :
on_false[result_index]`, where `pred_element = rank(pred) = 0 ? pred[] :
pred[result_index]`. For quantized types, performs
`dequantize_select_quantize(pred, on_true, on_false, type(result))`.

#### Inputs

| Label | Name       | Type                                  | Constraints |
|-------|------------|---------------------------------------|-------------|
| (I1)  | `pred`     | tensor of type `i1`                   | (C1)        |
| (I2)  | `on_true`  | tensor or per-tensor quantized tensor | (C1-C2)     |
| (I3)  | `on_false` | tensor or per-tensor quantized tensor | (C2)        |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C2)        |

#### Constraints

* (C1) `rank(pred) = 0 or shape(pred) = shape(on_true)`.
* (C2) `baseline_type(on_true) = baseline_type(on_false) = baseline_type(result)`.

#### Examples

```mlir
// %pred: [[false, true], [true, false]]
// %on_true: [[1, 2], [3, 4]]
// %on_false: [[5, 6], [7, 8]]
%result = "stablehlo.select"(%pred, %on_true, %on_false) : (tensor<2x2xi1>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// %result: [[5, 2], [3, 8]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/select.mlir)

### select_and_scatter

#### Semantics

Scatters the values from the `source` tensor using `scatter` based on the
outcome of `reduce_window` of the `input` tensor using `select` and produces
a `result` tensor.

The following diagram shows how elements in `result` are computed from
`operand` and `source` using a concrete example.

![select_and_scatter](images/spec/select_and_scatter.svg)

More formally:

* `selected_values = reduce_window_without_init(...)` with the following inputs:
  * `inputs = [operand].`
  * `window_dimensions`, `window_strides`, and `padding` which are used as is.
  * `base_dilations = windows_dilations = 1`.
  * `body` is defined as:

   ```python
   def body(arg0: tensor<E>, arg1: tensor<E>) -> tensor<E>:
     return select(arg0, arg1) ? arg0 : arg1;
   ```

   where `E = element_type(operand)`, and `reduce_window_without_init` works
   exactly like `reduce_window`, except that the `schedule` of the underlying
   `reduce` (see [reduce](#reduce)) doesn't include init values. It is currently
   unspecified what happens if the corresponding window doesn't have values
   ([#731](https://github.com/openxla/stablehlo/issues/731)).
* `result[result_index] = reduce([source_values], [init_value], [0], scatter)`
 where:
  * `source_values = [source[source_index] for source_index in
   source_indices]`.
  * `selected_index(source_index) = operand_index` if
   `selected_values[source_index]` has the `operand` element
   from `operand_index`.
  * `source_indices = [source_index for source_index in
   indices(source) if selected_index(source_index) = result_index]`.

#### Inputs

| Label | Name                | Type                                                | Constraints             |
|-------|---------------------|-----------------------------------------------------|-------------------------|
| (I1)  | `operand`           | tensor or per-tensor quantized tensor               | (C1-C4), (C6), (C8-C11) |
| (I2)  | `source`            | tensor or per-tensor quantized tensor               | (C1), (C2)              |
| (I3)  | `init_value`        | 0-dimensional tensor or per-tensor quantized tensor | (C3)                    |
| (I4)  | `window_dimensions` | 1-dimensional tensor constant of type `si64`        | (C2), (C4), (C5)        |
| (I5)  | `window_strides`    | 1-dimensional tensor constant of type `si64`        | (C2), (C6), (C7)        |
| (I6)  | `padding`           | 2-dimensional tensor constant of type `si64`        | (C2), (C8)              |
| (I7)  | `select`            | function                                            | (C9)                    |
| (I8)  | `scatter`           | function                                            | (C10)                   |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C11-C12)   |

#### Constraints

<!-- markdownlint-disable line-length -->
* (C1) `element_type(operand) = element_type(source)`.
* (C2) `shape(source) = num_windows` where:
  * `padded_operand_shape = padding[:, 0] + shape(operand) + padding[:, 1]`.
  * `is_empty_window = padded_operand_shape = 0 || window_dimensions > padded_operand_shape`.
  * `num_windows = is_empty_window ? 0 : floor((padded_operand_shape - window_dimensions) / window_strides) + 1`.
* (C3) `element_type(init_value) = element_type(operand)`.
* (C4) `size(window_dimensions) = rank(operand)`.
* (C5) `0 < window_dimensions`.
* (C6) `size(window_strides) = rank(operand)`.
* (C7) `0 < window_strides`.
* (C8) `shape(padding) = [rank(operand), 2]`.
* (C9) `select` has type `(tensor<E>, tensor<E>) -> tensor<i1>` where
       `E = element_type(operand)`.
* (C10) `scatter` has type `(tensor<E>, tensor<E>) -> tensor<E>` where
  `is_promotable(element_type(operand), E)`.
* (C11) `shape(operand) = shape(result)`.
* (C12) `element_type(result) = E`.
<!-- markdownlint-enable line-length -->

#### Examples

```mlir
// %operand: [[1, 5], [2, 5], [3, 6], [4, 4]]
// %source: [[5, 6], [7, 8]]
// %init_value: 0
%result = "stablehlo.select_and_scatter"(%operand, %source, %init_value) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.compare"(%arg0, %arg1) {
      comparison_direction = #stablehlo<comparison_direction GE>
    } : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "stablehlo.return"(%0) : (tensor<i1>) -> ()
}, {
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  window_dimensions = array<i64: 3, 1>,
  window_strides = array<i64: 2, 1>,
  padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
} : (tensor<4x2xi64>, tensor<2x2xi64>, tensor<i64>) -> tensor<4x2xi64>
// %result: [[0, 0], [0, 0], [5, 14], [7, 0]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/select_and_scatter.mlir)

### send

#### Semantics

Sends `inputs` to a channel `channel_id` and produces a `result` token.

If `is_host_transfer` is `true`, then the operation transfers data to the
host. Otherwise, it transfers data to another device. What this means is
implementation-defined. This flag duplicates the information provided in
`channel_type`, so in the future we are planning to only keep one of them
([#666](https://github.com/openxla/stablehlo/issues/666)).

#### Inputs

| Label | Name               | Type                                            | Constraints |
|-------|--------------------|-------------------------------------------------|-------------|
| (I1)  | `inputs`           | variadic number of tensors or quantized tensors |             |
| (I2)  | `token`            | `token`                                         |             |
| (I3)  | `channel_id`       | constant of type `si64`                         |             |
| (I4)  | `channel_type`     | enum of `DEVICE_TO_DEVICE` and `DEVICE_TO_HOST` | (C1)        |
| (I5)  | `is_host_transfer` | constant of type `i1`                           | (C1)        |

#### Outputs

| Name     | Type    |
|----------|---------|
| `result` | `token` |

#### Constraints

* (C1) `channel_type` is defined as:
  * `DEVICE_TO_HOST` if `is_host_transfer = true`,
  * `DEVICE_TO_DEVICE` otherwise.

#### Examples

```mlir
%result = "stablehlo.send"(%operand, %token) {
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
  is_host_transfer = true
} : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/send_recv.mlir)

### shift_left

#### Semantics

Performs element-wise left-shift operation on the `lhs` tensor by `rhs` number
of bits and produces a `result` tensor.

#### Inputs

| Label | Name  | Type                   | Constraints |
|-------|-------|------------------------|-------------|
| (I1)  | `lhs` | tensor of integer type | (C1)        |
| (I2)  | `rhs` | tensor of integer type | (C1)        |

#### Outputs

| Name     | Type                   | Constraints |
|----------|------------------------|-------------|
| `result` | tensor of integer type | (C1)        |

#### Constraints

* (C1) `type(lhs) = type(rhs) = type(result)`.

#### Examples

```mlir
// %lhs: [-1, 0, 1]
// %rhs: [1, 2, 3]
%result = "stablehlo.shift_left"(%lhs, %rhs): (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
// %result: [-2, 0, 8]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/shift_left.mlir)

### shift_right_arithmetic

#### Semantics

Performs element-wise arithmetic right-shift operation on the `lhs` tensor by
`rhs` number of bits and produces a `result` tensor.

#### Inputs

| Label | Name  | Type                   | Constraints |
|-------|-------|------------------------|-------------|
| (I1)  | `lhs` | tensor of integer type | (C1)        |
| (I2)  | `rhs` | tensor of integer type | (C1)        |

#### Outputs

| Name     | Type                   | Constraints |
|----------|------------------------|-------------|
| `result` | tensor of integer type | (C1)        |

#### Constraints

* (C1) `type(lhs) = type(rhs) = type(result)`.

#### Examples

```mlir
// %lhs: [-1, 0, 8]
// %rhs: [1, 2, 3]
%result = "stablehlo.shift_right_arithmetic"(%lhs, %rhs): (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
// %result: [-1, 0, 1]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/shift_right_arithmetic.mlir)

### shift_right_logical

#### Semantics

Performs element-wise logical right-shift operation on the `lhs` tensor by `rhs`
number of bits and produces a `result` tensor.

#### Inputs

| Label | Name  | Type                   | Constraints |
|-------|-------|------------------------|-------------|
| (I1)  | `lhs` | tensor of integer type | (C1)        |
| (I2)  | `rhs` | tensor of integer type | (C1)        |

#### Outputs

| Name     | Type                   | Constraints |
|----------|------------------------|-------------|
| `result` | tensor of integer type | (C1)        |

#### Constraints

* (C1) `type(lhs) = type(rhs) = type(result)`.

#### Examples

```mlir
// %lhs: [-1, 0, 8]
// %rhs: [1, 2, 3]
%result = "stablehlo.shift_right_logical"(%lhs, %rhs): (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
// %result: [9223372036854775807, 0, 1]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/shift_right_logical.mlir)

### sign

#### Semantics

Returns the sign of the `operand` element-wise and produces a `result` tensor.
More formally, for each element `x`, the semantics can be expressed using
Python syntax as follows:

```python
def sign(x):
  if is_integer(x):
    if compare(x, 0, LT, SIGNED): return -1
    if compare(x, 0, EQ, SIGNED): return 0
    return 1
  elif is_float(x):
    if is_nan(x): return NaN
    if compare(x, -0.0, EQ, FLOAT): return -0.0
    if compare(x, +0.0, EQ, FLOAT): return +0.0
    if compare(x, 0.0, LT, FLOAT): return -1.0
    return 1.0
  elif is_complex(x):
    if is_nan(real(x)) or is_nan(imag(x)): return (NaN, NaN)
    if compare(x, (0.0, 0.0), EQ, FLOAT): return (0.0, 0.0)
    return divide(x, convert(abs(x), type(x)))
```

For quantized types, performs
`dequantize_op_quantize(sign, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                                     | Constraints |
|-------|-----------|------------------------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of signed integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                                     | Constraints |
|----------|------------------------------------------------------------------------------------------|-------------|
| `result` | tensor of signed integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// Logical values: +NaN, -1.0, -0.0, +0.0, 1.0
// operand: [0x7FFFFFFFFFFFFFFF, -1.0, -0.0, 0.0, 1.0]
%result = "stablehlo.sign"(%operand) : (tensor<5xf64>) -> tensor<5xf64>
// Logical values: +NaN, -1.0, -0.0, +0.0, 1.0
// %result: [0x7FFFFFFFFFFFFFFF, -1.0, -0.0, 0.0, 1.0]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/sign.mlir)

### sine

#### Semantics

Performs element-wise sine operation on `operand` tensor and produces a `result`
tensor. Depending on the element type, does the following:

* For floats: `sin` from IEEE-754.
* For complex numbers: complex sine.
* For quantized types: `dequantize_op_quantize(sine, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [
//            [0.0, 1.57079632],       // [0, pi/2]
//            [3.14159265, 4.71238898] // [pi, 3pi/2]
//           ]
%result = "stablehlo.sine"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[0.0, 1.0], [0.0, -1.0]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/sine.mlir)

### slice

#### Semantics

Extracts a slice from the `operand` using statically-computed starting indices
and produces a `result` tensor. `start_indices` contain the starting indices of
the slice for each dimension, `limit_indices` contain the ending indices
(exclusive) for the slice for each dimension, and `strides` contain the strides
for each dimension.

More formally, `result[result_index] = operand[operand_index]` where
`operand_index = start_indices + result_index * strides`.

#### Inputs

| Label | Name            | Type                                         | Constraints      |
|-------|-----------------|----------------------------------------------|------------------|
| (I1)  | `operand`       | tensor or per-tensor quantized tensor        | (C1-C3), (C5)    |
| (I2)  | `start_indices` | 1-dimensional tensor constant of type `si64` | (C2), (C3), (C5) |
| (I3)  | `limit_indices` | 1-dimensional tensor constant of type `si64` | (C2), (C3), (C5) |
| (I4)  | `strides`       | 1-dimensional tensor constant of type `si64` | (C2), (C4)       |

#### Outputs

| Name     | Type                                  | Constraints |
|----------|---------------------------------------|-------------|
| `result` | tensor or per-tensor quantized tensor | (C1), (C5)  |

#### Constraints

* (C1) `element_type(operand) = element_type(result)`.
* (C2) `size(start_indices) = size(limit_indices) = size(strides) =
  rank(operand)`.
* (C3) `0 <= start_indices <= limit_indices <= shape(operand)`.
* (C4) `0 < strides`.
* (C5) `shape(result) = ceil((limit_indices - start_indices) / strides)`.

#### Examples

```mlir
// %operand: [
//            [0, 0, 0, 0],
//            [0, 0, 1, 1],
//            [0, 0, 1, 1]
//           ]
%result = "stablehlo.slice"(%operand) {
  start_indices = array<i64: 1, 2>,
  limit_indices = array<i64: 3, 4>,
  strides = array<i64: 1, 1>
} : (tensor<3x4xi64>) -> tensor<2x2xi64>
// % result: [
//            [1, 1],
//            [1, 1]
//           ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/slice.mlir)

### sort

#### Semantics

Sorts 1-dimensional slices of `inputs` along the dimension `dimension` together,
according to a `comparator` and produces `results`.

Unlike similar inputs in other operations, `dimension` allows negative values,
with the semantics described below. In the future, this may be disallowed
for consistency reasons
([#1377](https://github.com/openxla/stablehlo/issues/1377)).

If `is_stable` is true, then the sorting is stable, that is, relative order of
elements considered to be equal by the comparator is preserved. For the case
where there is a single input, two elements `e1` and `e2` are considered to be
equal by the comparator if and only if
`comparator(e1, e2) = comparator(e2, e1) = false`. See the formalization below
for how this generalizes to multiple inputs.

More formally, for all `result_index` in `index_space(results[0])`:

* `adjusted_dimension = dimension >= 0 ? dimension : rank(inputs[0]) + dimension`.
* `result_slice = [ri0, ..., :, ..., riR-1]` where `riN` are individual
  elements in `result_index`, and `:` is inserted at `adjusted_dimension`.
* `inputs_together = (inputs[0]..., ..., inputs[N-1]...)`.
* `results_together[result_slice] = sort(inputs_together[result_slice], comparator_together)`.
* where `sort` sorts a 1-dimensional slice in non-descending order expecting
  that `comparator_together` returns `true` if the left-hand side argument is
  less than the right-hand second argument.
* &#32;

  ```python
  def comparator_together(lhs_together, rhs_together):
    args = []
    for (lhs_el, rhs_el) in zip(lhs_together, rhs_together):
      args.append(lhs_el)
      args.append(rhs_el)
    return comparator(*args)
  ```

* `(results[0]..., ..., results[N-1]...) = results_together`.

#### Inputs

| Label | Name         | Type                                                       | Constraints |
|-------|--------------|------------------------------------------------------------|-------------|
| (I1)  | `inputs`     | variadic number of tensors or per-tensor quantized tensors | (C1-C5)     |
| (I2)  | `dimension`  | constant of type `si64`                                    | (C4)        |
| (I3)  | `is_stable`  | constant of type `i1`                                      |             |
| (I4)  | `comparator` | function                                                   | (C5)        |

#### Outputs

| Name      | Type                                                       | Constraints |
|-----------|------------------------------------------------------------|-------------|
| `results` | variadic number of tensors or per-tensor quantized tensors | (C2), (C3)  |

#### Constraints

* (C1) `0 < size(inputs)`.
* (C2) `type(inputs...) = type(results...)`.
* (C3) `same(shape(inputs...) + shape(results...))`.
* (C4) `-R <= dimension < R`, where `R = rank(inputs[0])`.
* (C5) `comparator` has type
  `(tensor<E1>, tensor<E1>, ..., tensor<EN-1>, tensor<EN-1>) -> tensor<i1>`,
  where `Ei = element_type(inputs[i])`.

#### Examples

```mlir
// %input0 = [[1, 2, 3], [3, 2, 1]]
// %input1 = [[3, 2, 1], [1, 2, 3]]
%result0, %result1 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>):
    %predicate = "stablehlo.compare"(%arg0, %arg1) {
      comparison_direction = #stablehlo<comparison_direction GT>
    } : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "stablehlo.return"(%predicate) : (tensor<i1>) -> ()
}) {
  dimension = 0 : i64,
  is_stable = true
} : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<2x3xi64>)
// %result0 = [[3, 2, 3], [1, 2, 1]]
// %result1 = [[1, 2, 1], [3, 2, 3]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/sort.mlir)

### sqrt

#### Semantics

Performs element-wise square root operation on `operand` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `squareRoot` from IEEE-754.
* For complex numbers: complex square root.
* For quantized types: `dequantize_op_quantize(sqrt, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [[0.0, 1.0], [4.0, 9.0]]
%result = "stablehlo.sqrt"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// %result: [[0.0, 1.0], [2.0, 3.0]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/sqrt.mlir)

### subtract

#### Semantics

Performs element-wise subtraction of two tensors `lhs` and `rhs` and produces a
`result` tensor. Depending on the element type, does the following:

* For integers: integer subtraction.
* For floats: `subtraction` from IEEE-754.
* For complex numbers: complex subtraction.
* For quantized types:
  * `dequantize_op_quantize(subtract, lhs, rhs, type(result))`.

#### Inputs

| Label | Name  | Type                                                                              | Constraints |
|-------|-------|-----------------------------------------------------------------------------------|-------------|
| (I1)  | `lhs` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |
| (I2)  | `rhs` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                              | Constraints |
|----------|-----------------------------------------------------------------------------------|-------------|
| `result` | tensor of integer, floating-point, or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(lhs) = baseline_type(rhs) = baseline_type(result)`.

#### Examples

```mlir
// %lhs: [[6, 8], [10, 12]]
// %rhs: [[5, 6], [7, 8]]
%result = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
// %result: [[1, 2], [3, 4]]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/subtract.mlir)

### tan

#### Semantics

Performs element-wise tangent operation on the `operand` tensor and produces a
`result` tensor. Depending on the element type, does the following:

* For floats: `tan` from IEEE-754.
* For complex numbers: complex tangent.
* For quantized types: `dequantize_op_quantize(tan, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [
//            [0.0, 1.57079632],       // [0, pi/2]
//            [3.14159265, 4.71238898] // [pi, 3pi/2]
//           ]
%result = "stablehlo.tan"(%operand) : (tensor<2x2xf64>) -> tensor<2x2xf64>
// %result: [
//           [0.0, 1.63312e+16],
//           [0.0, 5.44375e+15]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/tan.mlir)

### tanh

#### Semantics

Performs element-wise hyperbolic tangent operation on `operand` tensor and
produces a `result` tensor. Depending on the element type, does the following:

* For floats: `tanh` from IEEE-754.
* For complex numbers: complex hyperbolic tangent.
* For quantized types:
  * `dequantize_op_quantize(tanh, operand, type(result))`.

#### Inputs

| Label | Name      | Type                                                                    | Constraints |
|-------|-----------|-------------------------------------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_type(operand) = baseline_type(result)`.

#### Examples

```mlir
// %operand: [-1.0, 0.0, 1.0]
%result = "stablehlo.tanh"(%operand) : (tensor<3xf32>) -> tensor<3xf32>
// %result: [-0.76159416, 0.0, 0.76159416]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/tanh.mlir)

### transpose

#### Semantics

Permutes the dimensions of `operand` tensor using `permutation` and produces a
`result` tensor. More formally, `result[result_index] = operand[operand_index]`
where `result_index[d] = operand_index[permutation[d]]`.

#### Inputs

| Label | Name          | Type                                         | Constraints |
|-------|---------------|----------------------------------------------|-------------|
| (I1)  | `operand`     | tensor or quantized tensor                   | (C1-C4)     |
| (I2)  | `permutation` | 1-dimensional tensor constant of type `si64` | (C2-C4)     |

#### Outputs

| Name     | Type                       | Constraints   |
|----------|----------------------------|---------------|
| `result` | tensor or quantized tensor | (C1), (C3-C4) |

#### Constraints

* (C1) `element_type(result)` is given by:
  * `element_type(operand)`, if `!is_per_axis_quantized(operand)`.
  * `element_type(operand)` except that `quantization_dimension(operand)` and
    `quantization_dimension(result)` may differ, otherwise.
* (C2) `permutation` is a permutation of `range(rank(operand))`.
* (C3) `shape(result) = dim(operand, permutation...)`.
* (C4) If `is_per_axis_quantized(result)`, then
  `quantization_dimension(operand) =
  permutation(quantization_dimension(result))`.

#### Examples

```mlir
// %operand: [
//            [[1,2], [3,4], [5,6]],
//            [[7,8], [9,10], [11,12]]
//           ]
%result = "stablehlo.transpose"(%operand) {
  permutation = array<i64: 2, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// %result: [
//           [[1,7], [3,9], [5,11]],
//           [[2,8], [4,10], [6,12]]
//          ]
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/transpose.mlir)

### triangular_solve

#### Semantics

Solves batches of systems of linear equations with lower or upper triangular
coefficient matrices.

More formally, given `a` and `b`, `result[i0, ..., iR-3, :, :]` is the solution
to `op(a[i0, ..., iR-3, :, :]) * x = b[i0, ..., iR-3, :, :]` when `left_side` is
`true` or `x * op(a[i0, ..., iR-3, :, :]) = b[i0, ..., iR-3, :, :]` when
`left_side` is `false`, solving for the variable `x` where `op(a)` is determined
by `transpose_a`, which can be one of the following:

* `NO_TRANSPOSE`: Perform operation using `a` as-is.
* `TRANSPOSE`: Perform operation on transpose of `a`.
* `ADJOINT`: Perform operation on conjugate transpose of `a`.

Input data is read only from the lower triangle of `a`, if `lower` is `true` or
upper triangle of `a`, otherwise. Output data is returned in the same triangle;
the values in the other triangle are implementation-defined.

If `unit_diagonal` is true, then the implementation can assume that the diagonal
elements of `a` are equal to 1, otherwise the behavior is undefined.

For quantized types, performs
`dequantize_op_quantize(lambda x, y: triangular_solve(x, y, left_side, lower,
unit_diagonal, transpose_a), a, b, type(result))`.

#### Inputs

| Label | Name            | Type                                                                    | Constraints |
|-------|-----------------|-------------------------------------------------------------------------|-------------|
| (I1)  | `a`             | tensor of floating-point or complex type or per-tensor quantized tensor | (C1-C3)     |
| (I2)  | `b`             | tensor of floating-point or complex type or per-tensor quantized tensor | (C1-C4)     |
| (I3)  | `left_side`     | constant of type `i1`                                                   | (C3)        |
| (I4)  | `lower`         | constant of type `i1`                                                   |             |
| (I5)  | `unit_diagonal` | constant of type `i1`                                                   |             |
| (I6)  | `transpose_a`   | enum of `NO_TRANSPOSE`, `TRANSPOSE`, and `ADJOINT`                      |             |

#### Outputs

| Name     | Type                                                                    | Constraints |
|----------|-------------------------------------------------------------------------|-------------|
| `result` | tensor of floating-point or complex type or per-tensor quantized tensor | (C1)        |

#### Constraints

* (C1) `baseline_element_type(a) = baseline_element_type(b)`.
* (C2) `2 <= rank(a) = rank(b) = R`.
* (C3) The relationship between `shape(a)` and `shape(b)` is defined as follows:
  * `shape(a)[:-3] = shape(b)[:-3]`.
  * `dim(a, -2) = dim(a, -1) = dim(b, left_side ? -2 : -1)`.
* (C4) `baseline_type(b) = baseline_type(result)`.

#### Examples

```mlir
// %a = [
//       [1.0, 0.0, 0.0],
//       [2.0, 4.0, 0.0],
//       [3.0, 5.0, 6.0]
//      ]
// %b = [
//       [2.0, 0.0, 0.0],
//       [4.0, 8.0, 0.0],
//       [6.0, 10.0, 12.0]
//      ]
%result = "stablehlo.triangular_solve"(%a, %b) {
  left_side = true,
  lower = true,
  unit_diagonal = false,
  transpose_a = #stablehlo<transpose NO_TRANSPOSE>
} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// %result: [
//           [2.0, 0.0, 0.0],
//           [0.0, 2.0, 0.0],
//           [0.0, 0.0, 2.0]
//          ]
```

### tuple

> Note: Per [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283),
> this op is being explored for deprecation as it appears to be unused by both
> frameworks and compilers. As such, it has limited compatibility guarantees
> (6 months).

#### Semantics

Produces a `result` tuple from values `val`.

#### Inputs

| Label | Name  | Type                      | Constraints |
|-------|-------|---------------------------|-------------|
| (I1)  | `val` | variadic number of values | (C1)        |

#### Outputs

| Name     | Type  | Constraints |
|----------|-------|-------------|
| `result` | tuple | (C1)        |

#### Constraints

* (C1) `result` has type `tuple<E0, ..., EN-1>` where `Ei = type(val[i])`.

#### Examples

```mlir
// %val0: [1.0, 2.0]
// %val1: (3)
%result = "stablehlo.tuple"(%val0, %val1) : (tensor<2xf32>, tuple<tensor<i32>>) -> tuple<tensor<2xf32>, tuple<tensor<i32>>>
// %result: ([1.0, 2.0], (3))
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/tuple_and_get_tuple_element.mlir)

### uniform_dequantize

#### Semantics

Performs element-wise conversion of quantized tensor `operand` to a
floating-point tensor `result` according to the quantization parameters defined
by the `operand` type.

More formally, `result = dequantize(operand)`.

#### Inputs

| Label | Name      | Type             | Constraints |
|-------|-----------|------------------|-------------|
| (I1)  | `operand` | quantized tensor | (C1), (C2)  |

#### Outputs

| Name     | Type                          | Constraints |
|----------|-------------------------------|-------------|
| `result` | tensor of floating-point type | (C1), (C2)  |

#### Constraints

* (C1) `shape(operand) = shape(result)`.
* (C2) `element_type(result) = expressed_type(operand)`.

#### Examples

```mlir
// %operand: [10, 10]
%result = "stablehlo.uniform_dequantize"(%operand) : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2xf32>
// %result: [4.0, 15.0]
```

### uniform_quantize

#### Semantics

Performs element-wise conversion of floating-point tensor or quantized tensor
`operand` to a quantized tensor `result` according to the quantization
parameters defined by the `result` type.

More formally,

* If `is_float(operand)`:
  * `result = quantize(operand, type(result))`.
* If `is_quantized(operand)`:
  * `float_result = dequantize(operand)`.
  * `result = quantize(float_result, type(result))`.

#### Inputs

| Label | Name      | Type                                       | Constraints |
|-------|-----------|--------------------------------------------|-------------|
| (I1)  | `operand` | tensor of floating-point or quantized type | (C1), (C2)  |

#### Outputs

| Name     | Type             | Constraints |
|----------|------------------|-------------|
| `result` | quantized tensor | (C1), (C2)  |

#### Constraints

* (C1) `shape(operand) = shape(result)`.
* (C2) `expressed_type(result) = is_float(operand) ? element_type(operand) :
  expressed_type(operand)`.

#### Examples

```mlir
// %operand: [4.0, 15.0]
%result = "stablehlo.uniform_quantize"(%operand) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>
// %result: [10, 10]

// %operand: [10, 10]
%result = "stablehlo.uniform_quantize"(%operand) : (tensor<2x!quant.uniform<i8:f32:0, {0.1:-30,0.5:-20}>>) -> tensor<2x!quant.uniform<i8:f32:0, {0.1:-20,0.2:-30}>>
// %result: [20, 45]
```

### while

#### Semantics

Produces the output from executing `body` function 0 or more times while the
`cond` function outputs `true`. More formally, the semantics can be expressed
using Python syntax as follows:

```python
internal_state = operand
while cond(*internal_state):
  internal_state = body(*internal_state)
results = internal_state
```

The behavior of an infinite loop is TBD
([#383](https://github.com/openxla/stablehlo/issues/383)).

#### Inputs

| Label | Name      | Type                                                    | Constraints |
|-------|-----------|---------------------------------------------------------|-------------|
| (I1)  | `operand` | variadic number of tensors, quantized tensors or tokens | (C1-C3)     |
| (I2)  | `cond`    | function                                                | (C1)        |
| (I3)  | `body`    | function                                                | (C2)        |

#### Outputs

| Name      | Type                                                    | Constraints |
|-----------|---------------------------------------------------------|-------------|
| `results` | variadic number of tensors, quantized tensors or tokens | (C3)        |

#### Constraints

* (C1) `cond` has type `(T0, ..., TN-1) -> tensor<i1>`, where
       `Ti = type(operand[i])`.
* (C2) `body` has type `(T0, ..., TN-1) -> (T0, ..., TN-1)`, where
       `Ti = type(operand[i])`.
* (C3) `type(results...) = type(operand...)`.

#### Examples

```mlir
// %init_i: 1
// %init_sum: 0
// %one: 1
// %ten: 10
%results0, %results1 = "stablehlo.while"(%init_i, %init_sum) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %cond = "stablehlo.compare"(%arg0, %ten) {
      comparison_direction = #stablehlo<comparison_direction LT>
    } : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cond : tensor<i1>
  }, {
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %new_sum = stablehlo.add %arg1, %one : tensor<i64>
    %new_i = stablehlo.add %arg0, %one : tensor<i64>
    stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
}) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
// %results0: 10
// %results1: 10
```

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/while.mlir)

### xor

#### Semantics

Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a `result`
tensor. Depending on the element type, does the following:

* For booleans: logical XOR.
* For integers: bitwise XOR.

#### Inputs

| Label | Name  | Type                              | Constraints |
|-------|-------|-----------------------------------|-------------|
| (I1)  | `lhs` | tensor of boolean or integer type | (C1)        |
| (I2)  | `rhs` | tensor of boolean or integer type | (C1)        |

#### Outputs

| Name     | Type                              | Constraints |
|----------|-----------------------------------|-------------|
| `result` | tensor of boolean or integer type | (C1)        |

#### Constraints

* (C1) `type(lhs) = type(rhs) = type(result)`.

#### Examples

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

&nbsp;[More Examples](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret/xor.mlir)

## Dialect Interop

At the moment, StableHLO programs in the wild sometimes contain operations that
are not defined by StableHLO.

### Module, Function, Call and Return

StableHLO uses upstream MLIR operations for ModuleOp, FuncOp, CallOp, and
ReturnOp. This was done for better interop with existing MLIR machinery, as many
useful passes are written targeting FuncOp and ModuleOp, and many compilation
pipelines expect these ops to be present.  Full compatibility guarantees are
applied to these ops. If anything ever changes about these ops in an
incompatible way (i.e. removal), StableHLO equivalents will be added to preserve
compatibility.

### CHLO

The CHLO opset contains higher level operations that decompose to StableHLO.
Currently there are no compatibility guarantees for CHLO. For compatibility
guarantees, the [chlo-legalize-to-stablehlo pass](https://github.com/openxla/stablehlo/blob/12fd0a9e7b3c6f3dea3defc513870c962e62726d/stablehlo/transforms/Passes.td#L119)
must be used prior to serialization.

### Shape Operations

It is a common use case in the community to use certain operations from core
MLIR dialects in dynamic StableHLO programs to perform shape computations.
Most commonly, these include [`shape` dialect](https://mlir.llvm.org/docs/Dialects/ShapeDialect/)
ops like `shape_of` or `num_elements`, [`tensor` dialect](https://mlir.llvm.org/docs/Dialects/TensorOps/)
ops like `dim` or `from_elements`, and the builtin `index` type.

The [Dynamism RFC > O2](https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md#o2)
denotes these as out of scope, however some support for `index` types is
included for interop purposes. There are no compatibility guarantees for these
ops or types. The [shape-legalize-to-stablehlo](https://github.com/openxla/stablehlo/blob/12fd0a9e7b3c6f3dea3defc513870c962e62726d/stablehlo/transforms/Passes.td#L136)
pass can be used to convert these operations to fully supported StableHLO ops.

## Deprecated Operations

There are several StableHLO operations that were inherited from
[MHLO](https://github.com/openxla/xla/blob/d63deb9250b9c212445290bd08c6effb5b6d0a2b/xla/mlir_hlo/mhlo/IR/hlo_ops.td)
which are deprecated and on the way out of StableHLO. The full details on these
removals can be found in the [StableHLO v1.0 Cleanup #2283](https://github.com/openxla/stablehlo/pull/2283).
The tracker issue for these deprecations is [#2340](https://github.com/openxla/stablehlo/issues/2340).

These operations fall into a few categories:

* "Not in HLO" category of StableHLO operations - they were initially part of
  the StableHLO opset but have been later deemed to not fit it well:
  `broadcast`, `create_token`, `cross-replica-sum`, `dot`, `einsum`,
  `torch_index_select`, `unary_einsum`
  ([#3](https://github.com/openxla/stablehlo/issues/3)).
* Unused ops - These operations may have been useful at some point, but the ops
  were either underdeveloped, or the pipelines using these ops have been
  refactored to not require them anymore. This includes `map`, `tuple` ([#598](https://github.com/openxla/stablehlo/issues/598)),
  `get_tuple_element`, `rng`, `complex` comparisons [#560](https://github.com/openxla/stablehlo/issues/560),
  and convolution `window_reversal` ([#1181](https://github.com/openxla/stablehlo/issues/1181)).

Some of these ops can be removed easily given that they can be expressed using
existing ops (`broadcast`, `create_token`, `cross-replica-sum`, `dot`,
`unary_einsum`) and will be removed after the existing compatibilty window
passes (6 months). Others are still being explored for removal (`einsum`,
`get_tuple_element`, `map`, `rng` `torch_index_select`, `tuple`, `complex`
comparisons, `window_reversal`). Pending community feedback,
these ops will either be removed, or added to the spec with full support. Until
these ops futures are known, they are only guaranteed 6 months of compatibility.

## Execution

### Sequential execution

A StableHLO program is executed by providing input values to the `main` function
and computing output values. Output values of a function are computed by
executing the graph of ops rooted in the corresponding `return` op.

The execution order is implementation-defined as long as it is aligned with
dataflow, i.e. if ops are executed before their uses. In StableHLO, all
side-effecting ops consume one token and produce one token (multiple tokens can
be multiplexed into one token via `after_all`), so the execution order of side
effects is also aligned with dataflow. For example, in the below program
there are two possible execution orders: `%0` → `%1` → `%2` → `return` and
`%1` → `%0` → `%2` → `return`.

```mlir
func.func @main() -> tensor<f64> {
  %0 = stablehlo.constant dense<1.0> : tensor<f64>
  %1 = stablehlo.constant dense<2.0> : tensor<f64>
  %2 = stablehlo.add %0, %1 : tensor<f64>
  return %2 : tensor<f64>
}
```

More formally, a **StableHLO process** is a combination of:
1\) a StableHLO program, 2) operation statuses (not executed yet,
already executed), and 3) intermediate values that the process is working on.
The process starts with input values to the `main` function, progresses through
the graph of ops updating operation statuses and intermediate values and
finishes with output values. Further formalization is TBD
([#484](https://github.com/openxla/stablehlo/issues/484)).

### Parallel execution

StableHLO programs can be executed in parallel, organized into a 2D process grid
of `num_replicas` by `num_partitions` which both have type `ui32`.

In the **StableHLO process grid**, `num_replicas * num_partitions` of StableHLO
processes are executing at the same time. Each process has a unique
`process_id = (replica_id, partition_id)`, where
`replica_id` in `replica_ids = range(num_replicas)` and
`partition_id` in `partition_ids = range(num_partitions)` which both have
type `ui32`.

The size of the process grid is known statically for every program (in the
future, we are planning to make it an explicit part of StableHLO programs
[#650](https://github.com/openxla/stablehlo/issues/650)), and the position
within the process grid is known statically for every process. Each process has
access to its position within the process grid via the `replica_id` and
`partition_id` ops.

Within the process grid, the programs can all be the same (in the "Single
Program, Multiple Data" style), can all be different (in the "Multiple Program,
Multiple Data" style) or something in between. In the future, we are planning
to introduce support for other idioms of defining parallel StableHLO programs,
including GSPMD ([#619](https://github.com/openxla/stablehlo/issues/619)).

Within the process grid, the processes are mostly independent from each other -
they have separate operation statuses, separate input/intermediate/output values
and most of the ops are executed separately between processes, with the
exception of a small number of collective ops described below.

Given that execution of most of the ops is only using values from the same
process, it is usually unambiguous to refer to these values by their names.
However, when describing semantics of collective ops, that is insufficient, and
that gives rise to the notation `name@process_id` to refer to the value `name`
within a particular process. (From that perspective, unqualified `name` can be
viewed as a shorthand for `name@(replica_id(), partition_id())`).

The execution order across processes is implementation-defined, except for the
synchronization introduced by point-to-point communication and collective ops
as described below.

### Point-to-point communication

StableHLO processes can communicate with each other through
**StableHLO channels**. A channel is represented by a positive id of type
`si64`. Through various ops, it is possible to send values to channels and
receive them from channels.

Further formalization, e.g. where these channel ids are coming from, how
processes programs become aware of them and what kind of synchronization is
introduced by them, is TBD
([#484](https://github.com/openxla/stablehlo/issues/484)).

### Streaming communication

Every StableHLO process has access to two streaming interfaces:

* **Infeed** that can be read from.
* **Outfeed** that can be written to.

Unlike channels, which are used to communicate between processes and therefore
have processes at both of their ends, infeeds and outfeeds have their other
end implementation-defined.

Further formalization, e.g. how streaming communication influences execution
order and what kind of synchronization is introduced by it, is TBD
([#484](https://github.com/openxla/stablehlo/issues/484)).

### Collective ops

There are six collective ops in StableHLO: `all_gather`, `all_reduce`,
`all_to_all`, `collective_broadcast`, `collective_permute`, and
`reduce_scatter`. All these ops split the processes in the StableHLO process
grid into **StableHLO process groups** and execute a joint computation within
each process group, independently from other process groups.

Within each process group, collective ops may introduce a synchronization
barrier. Further formalization, e.g. elaborating on when exactly this
synchronization happens, how exactly the processes arrive at this barrier,
and what happens if they don't, is TBD
([#484](https://github.com/openxla/stablehlo/issues/484)).

If the process group involves cross-partition communication, i.e. there are
processes in the process group whose partition ids are different, then execution
of the collective op needs a channel, and the collective op must provide a
positive `channel_id` of type `si64`. Cross-replica communication doesn't need
channels.

The computations performed by the collective ops are specific to individual ops
and are described in individual op sections above. However, the strategies by
which the process grid is split into process groups are shared between these ops
and are described in this section. More formally, StableHLO supports the
following four strategies.

#### cross_replica

Only cross-replica communications happen within each process group. This
strategy takes `replica_groups` - a list of lists of replica ids - and computes
a Cartesian product of `replica_groups` by `partition_ids`. `replica_groups`
must have unique elements and cover all `replica_ids`. More formally, using
Python syntax:

```python
def cross_replica(replica_groups: List[List[ReplicaId]]) -> List[List[ProcessId]]:
  for replica_group in replica_groups:
    for partition_id in partition_ids:
      process_group = []
      for replica_id in replica_group:
        process_group.append((replica_id, partition_id))
      yield process_group
```

For example, for `replica_groups = [[0, 1], [2, 3]]` and `num_partitions = 2`,
`cross_replica` will produce
`[[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(2, 0), (3, 0)], [(2, 1), (3, 1)]]`.

#### cross_partition

Only cross-partition communications happen within each process group. This
strategy takes `partition_groups` - a list of lists of partition ids - and
computes a Cartesian product of `partition_groups` by `replica_ids`.
`partition_groups` must have unique elements and cover all `partition_ids`.
More formally, using Python syntax:

```python
def cross_partition(partition_groups: List[List[PartitionId]]) -> List[List[ProcessId]]:
  for partition_group in partition_groups:
    for replica_id in replica_ids:
      process_group = []
      for partition_id in partition_group:
        process_group.append((replica_id, partition_id))
      yield process_group
```

For example, for `partition_groups = [[0, 1]]` and `num_replicas = 4`,
`cross_partition` will produce
`[[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)]]`.

#### cross_replica_and_partition

Both cross-replica and cross-partition communications may happen within each
process group. This strategy takes `replica_groups` - a list of lists of
replica ids - and computes Cartesian products of each `replica_group` by
`partition_ids`. `replica_groups` must have unique elements and cover all
`replica_ids`. More formally, using Python syntax:

```python
def cross_replica_and_partition(replica_groups: List[List[ReplicaId]]) -> List[List[ProcessId]]:
  for replica_group in replica_groups:
    process_group = []
    for partition_id in partition_ids:
      for replica_id in replica_group:
        process_group.append((replica_id, partition_id))
    yield process_group
```

For example, for `replica_groups = [[0, 1], [2, 3]]` and `num_partitions = 2`,
`cross_replica_and_partition` will produce
`[[(0, 0), (1, 0), (0, 1), (1, 1)], [(2, 0), (3, 0), (2, 1), (3, 1)]]`.

#### flattened_ids

This strategy takes `flattened_id_groups` - a list of lists of "flattened"
process ids in the form of `replica_id * num_partitions + partition_id` - and
turns them into process ids. `flattened_id_groups` must have unique elements
and cover all `process_ids`. More formally, using Python syntax:

```python
def flattened_ids(flattened_id_groups: List[List[ui32]]) -> List[List[ProcessId]]:
  for flattened_id_group in flattened_id_groups:
    process_group = []
    for flattened_id in flattened_id_group:
      replica_id = flattened_id // num_partitions
      partition_id = flattened_id % num_partitions
      process_group.append((replica_id, partition_id))
    yield process_group
```

For example, for `flattened_id_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]`,
`num_replicas = 4` and `num_partitions = 2`, `flattened_ids` will produce
`[[(0, 0), (0, 1), (1, 0), (1, 1)], [(2, 0), (2, 1), (3, 0), (3, 1)]]`.

### Accuracy

At the moment, StableHLO does not provide guarantees about numerical accuracy,
but this may change in the future
([#1156](https://github.com/openxla/stablehlo/issues/1156)).

### Execution semantics of quantized operation

The interpretation of quantized StableHLO operations may vary depending on the
hardware requirements and capabilities. For instance, some hardware may opt to
interpret quantized operations using a "dequantize, perform floating-point
operation, and finally quantize" strategy. Others may perform the entire
computation with integer arithmetic. Consequently, the interpretation of
quantized StableHLO operations is exclusively determined by the specific
implementation. The interpretation of hybrid quantization
([#1575](https://github.com/openxla/stablehlo/issues/1575)) should be based on
the it's semantics as prescribed in the specification (via
[1792](https://github.com/openxla/stablehlo/pull/1792)).

### Errors

StableHLO programs are validated through an extensive set of constraints for
individual ops, which rules out many classes of errors prior to run time.
However, error conditions are still possible, e.g. through integer overflows,
out-of-bounds accesses, etc. Unless explicitly called out, all these errors
result in implementation-defined behavior, but this may change in the
future ([#1157](https://github.com/openxla/stablehlo/issues/1157)).

#### Floating-point exceptions

As an exception to this rule, floating-point exceptions in StableHLO programs
have well-defined behavior. Operations which result in exceptions defined by the
IEEE-754 standard (invalid operation, division-by-zero, overflow, underflow, or
inexact exceptions) produce default results (as defined in the standard) and
continue execution without raising the corresponding status flag; similar to
`raiseNoFlag` exception handling from the standard. Exceptions for nonstandard
operations (e.g. complex arithmetic and certain transcendental functions) are
implementation-defined.

#### Shape mismatches

StableHLO supports dynamically-shaped tensors. However, shapes have to agree at
runtime, otherwise the behavior is undefined. StableHLO does not explicitly
provide an op that can assert that a tensor has a given shape at runtime.
Generating correct code is the responsibility of the producer.

As a specific example, the below program is valid. However, at runtime, the
exact shapes of `%arg0` and `%arg1` will have to be the same, otherwise the
behavior of the program is undefined:

```mlir
func.func @foo(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<?xi32>
    return %0 : tensor<?xi32>
}
```

## Notation

For describing syntax, this document is using the modified ISO flavor of EBNF
syntax ([ISO/IEC 14977:1996](https://www.iso.org/standard/26153.html),
[Wikipedia](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form)),
with two modifications: 1) rules are defined using `::=` rather than `=`,
2) concatenation is expressed using juxtaposition rather than `,`.

For describing semantics (i.e. within "Types", "Constants" and "Ops" sections),
we are using formulas which are based on Python syntax extended with support
for concisely expressing array operations as described below. This works well
for small snippets of code, but in rare cases when larger snippets of code are
needed, we use vanilla Python syntax which is always introduced explicitly.

### Formulas

Let's explore how formulas work based on an example from the `dot_general`
specification. One of the constraints for this operation looks as follows:
`dim(lhs, lhs_batching_dimensions...) = dim(rhs, rhs_batching_dimensions...)`.

The names used in this formula come from two sources: 1) global functions,
i.e. `dim`, 2) member definitions of the corresponding program element, i.e.
`lhs`, `lhs_batching_dimensions`, `rhs` and `rhs_batching_dimensions` inputs
defined in the "Inputs" section of `dot_general`.

As mentioned above, the syntax of this formula is Python-based with some
conciseness-oriented extensions. To make sense of the formula, let's transform
it into vanilla Python syntax.

A) In these formulas, we are using `=` to represent equality, so the first step
towards obtaining Python syntax is replacing `=` with `==`, as follows:
`dim(lhs, lhs_batching_dimensions...) == dim(rhs, rhs_batching_dimensions...)`.

B) Also, these formulas support ellipses (`...`) which turn scalar expressions
into tensor expressions. In a nutshell, `f(xs...)` roughly means "for each
scalar `x` in the tensor `xs`, compute a scalar `f(x)` and then return all
these scalar results together as a tensor result". In vanilla Python syntax,
our example formula turns into:
`[dim(lhs, dim1) for dim1 in lhs_batching_dimensions] ==
[dim(rhs, dim2) for dim2 in rhs_batching_dimensions]`.

Thanks to ellipses, it is often possible to avoid working at the level of
individual scalars. However, in some tricky cases, lower-level semi-informal
syntax may be used like in the `start_indices[bi0, ..., :, ..., biN]` formula
from the `gather` specification. In the service of conciseness, we don't
provide an exact formalism for translating such syntax to vanilla Python, in
hopes that it is still intuitively understandable on case-by-case basis.
Please let us know if some specific formulas look opaque, and we'll try to
improve them.

Also, you will notice that formulas use ellipses to expand all sorts of lists,
including tensors, lists of tensors (which e.g. can arise from a variadic
number of tensors), etc. This is another area where we don't provide an exact
formalism (e.g. lists are not even part of the StableHLO type system) and
instead rely on intuitive understandability.

C) The final noteworthy notational vehicle that we employ is implicit
broadcasting. While the StableHLO opset doesn't support implicit broadcasting,
the formulas do, also in the service of conciseness. In a nutshell, if a scalar
is used in a context where a tensor is expected, the scalar is broadcasted to
the expected shape.

To continue the `dot_general` example, here's another constraint:
`0 <= lhs_batching_dimensions < rank(lhs)`. As defined in the `dot_general`
specification, `lhs_batching_dimensions` is a tensor, however both `0` and
`rank(lhs)` are scalars. After we apply implicit broadcasting, the formula will
become `[0, ..., 0] <= lhs_batching_dimensions < [rank(lhs), ..., rank(lhs)]`.

When applied to a particular `dot_general` operation, this formula will
evaluate to a tensor of booleans. When formulas are used as constraints, the
constraint holds if the formula evaluates to either `true` or to a tensor which
only has `true` elements.

### Names

In formulas, lexical scope includes: 1) global functions, 2) member definitions,
3) local definitions. The list of global functions is provided below. The list
of element definitions depends on the program element that the notation is
applied to:

* For operations, member definitions include names introduced in "Inputs" and
  "Outputs" sections.
* For everything else, member definitions include structural parts of the
  program element, named after the corresponding EBNF non-terminals. Most of
  the time, the names of these structural parts are obtained by converting the
  names of the non-terminals to snake case (e.g. `IntegerLiteral` =>
  `integer_literal`), but sometimes names get abbreviated in the process (e.g.
  `QuantizationStorageType` => `storage_type`) in which case the names are
  introduced explicitly similarly to "Inputs" / "Outputs" sections in operation
  specifications.
* Additionally, member definitions always include `self` to refer to the
  corresponding program element.

### Values

When formulas are evaluated, they work with the following types of values:
1\) `Value` (actual values, e.g. `dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>`;
they always know their types),
2\) `Placeholder` (future values, e.g. `lhs`, `rhs` or `result`; their actual
values are not known yet, only their types are known),
3\) `Type` (types as defined in the "Types" section),
4\) `Function` (global functions as defined in the "Functions" section).

Depending on the context, names may be referring to different values. More
specifically, the "Semantics" section for ops (and equivalents for other program
elements) defines runtime logic, so all inputs are available as `Value`.
In contrast, the "Constraints" section for ops (and equivalents) defines
"compile-time" logic, i.e. something that is typically executed before runtime,
so only constant inputs are available as `Value` and other inputs are
available only as `Placeholder`.

| Names               | In "Semantics"            | In "Constraints"          |
|---------------------|---------------------------|---------------------------|
| Global functions    | `Function`                | `Function`                |
| Constant inputs     | `Value`                   | `Value`                   |
| Non-constant inputs | `Value`                   | `Placeholder`             |
| Outputs             | `Value`                   | `Placeholder`             |
| Local definitions   | Depends on the definition | Depends on the definition |

Let's consider an example `transpose` operation:

```mlir
%result = "stablehlo.transpose"(%operand) {
  permutation = dense<[2, 1, 0]> : tensor<3xi64>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
```

For this operation, `permutation` is a constant, so it's available as a `Value`
in both semantics and constraints. In contrast, `operand` and `result` are
available as a `Value` in semantics but only as a `Placeholder` in constraints.

### Functions

#### Construction of types

There are no functions that can be used to construct types. Instead, we directly
use type syntax because it's typically more concise. E.g.
`(tensor<E>, tensor<E>) -> (tensor<E>)` rather than `function_type(
[tensor_type([], E), tensor_type([], E)], [tensor_type([], E)])`.

#### Functions on types

* `element_type` is defined on tensor types and quantized tensor types and
returns, respectively, the `TensorElementType` or `QuantizedTensorElementType`
part of the corresponding `TensorType` or `QuantizedTensorType`.

```python
def element_type(x: Value | Placeholder | Type):
 if type(x) == TensorType:
    return tensor_element_type(x)
  if type(x) == QuantizedTensorType:
    return quantized_tensor_element_type(x)
  if type(x) is not Type:
    return element_type(type(x))
```

* `is_per_axis_quantized(x: Value | Placeholder | Type) -> Value` is a shortcut
for `is_quantized(x) and quantization_dimension(x) is not None`.

* `is_per_tensor_quantized(x: Value | Placeholder | Type) -> Value` is a
shortcut for `is_quantized(x) and quantization_dimension(x) is None`.

* `is_promotable(x: Type, y: Type) -> bool` checks if type `x` can be promoted
to type `y`.  When `x` and `y` are `QuantizedTensorElementType`s, the promotion
is applied only to the `storage_type`. This specific version of promotion is
currently used in context of reduction computation (refer to
[RFC](https://github.com/openxla/stablehlo/pull/1664) for more details).

```python
def is_promotable(x: Type, y: Type) -> Value:
  is_same_type = (is_bool(x) and is_bool(y)) or
    (is_integer(x) and is_integer(y)) or (is_float(x) and is_float(y)) or
    (is_complex(x) and is_complex(y)) or
    (is_quantized(x) and is_quantized(y) and expressed_type(x) = expressed_type(y))

  if is_same_type == False:
    return False

  if is_integer(x) or is_float(x):
    return bitwidth(x) <= bitwidth(y)

  if is_complex(x):
    return bitwidth(element_type(x)) <= bitwidth(element_type(y))

  if is_quantized(x):
    return bitwidth(storage_type(x)) <= bitwidth(storage_type(y))

  return false
```

* `is_quantized(x: Value | Placeholder | Type) -> Value` is a shortcut for
`is_quantized_tensor_element_type(x)`.

* `is_type_name(x: Value | Placeholder | Type) -> Value`. Available for all
types. For example, `is_float(x)` returns `true` if `x` is a `FloatType`.
If `x` is a value or placeholder, this function is a shortcut for
`is_type_name(type(x))`.

* `max_value(x: Type) -> Value` returns the maximum value of an
`TensorElementType`.  If `x` is not an `TensorElementType`, returns `None`.

* `min_value(x: Type) -> Value` returns the minimum possible value of an
`TensorElementType`. If `x` is not an `TensorElementType`, returns `None`.

* `member_name(x: Value | Placeholder | Type) -> Any`. Available for all member
definitions `member_name` of all types. For example, `tensor_element_type(x)`
returns the `TensorElementType` part of a corresponding `TensorType`.
If `x` is a value or placeholder, this function is a shortcut for
`member_name(type(x))`.  If `x` is not a type that has an appropriate member, or
a value or a placeholder of such a type, returns `None`.

* `is_empty_algorithm(*args: Type)` checks if all dot algorithm fields are set
  to `None`. This is needed since dot algorithms have implementation defined
  default behaviors, so specifying a default value would be incorrect.

#### Construction of values

* `operation_name(*xs: Value | Type) -> Value`. Available for all operations.
For example, `add(lhs, rhs)` takes two tensor values `lhs` and `rhs` and
returns the output of evaluating the `add` operation with these inputs.
For some operations e.g. `broadcast_in_dim`, types of their outputs are
"load-bearing", i.e. needed to evaluate an operation. In this case, the function
takes these types as arguments.

#### Functions on values

* All Python's operators and functions are available. E.g. both
[subscription](https://docs.python.org/3/reference/expressions.html#subscriptions)
and [slicing](https://docs.python.org/3/reference/expressions.html#slicings)
notations from Python are available to index into tensors, quantized tensors
and tuples.

* `to_destination_type(x: Value, destination_type: Type) -> Value` is defined on
tensors and returns the converted value of `x` based on the `type(x)` and
`destination_type` as follows:

```python
def to_destination_type(x: Value, destination_type: Type) -> Value:
  if type(x) == destination_type:
    return x

  if is_quantized(destination_type):
    if is_quantized(type(x)):
      return quantize(x, destination_type)
    assert is_float(type(x))
    return quantize(x, destination_type)

  if is_quantized(type(x)):
    assert destination_type = expressed_type(type(x))
    return dequantize(type(x))

  return convert(x, destination_type)
```

There is early discussion on merging `convert`, `uniform_quantize` and
`uniform_dequantize` operations ([#1576](https://github.com/openxla/stablehlo/issues/1576)).
After the merge we do not need the above function and can use the operation name
for `convert` instead.

* `is_nan(x: Value) -> Value` is defined on tensors and returns `true` if
all elements of `x` are `NaN` or `false` otherwise. If `x` is not a tensor,
returns `None`.

* `is_sorted(x: Value) -> Value` is defined on tensors and returns `true` if
elements of `x` are sorted in ascending order with respect to the ascending
lexicographical order of their indices or `false` otherwise. If `x` is not a
tensor, returns `None`.

* `is_unique(x: Value) -> Value` is defined on tensors and returns `true` if `x`
doesn't have duplicate elements or `false` otherwise. If `x` is not a tensor,
returns `None`.

* `member_name(x: Value) -> Any` is defined for all member definitions
`member_name` of all values. For example, `real_part(x)` returns the `RealPart`
part of a corresponding `ComplexConstant`. If `x` is not a value that has an
appropriate member, returns `None`.

* `same(x: Value) -> Value` is defined on tensors and returns `true` if
elements of `x` are all equal to each other or `false` otherwise. If the tensor
doesn't have elements, that counts as "all equal to each other", i.e. the
function returns `true`. If `x` is not a tensor, returns `None`.

* `split(x: Value, num_results: Value, axis: Value) -> Value` is defined on
tensors and returns `num_results` slices of `x` along the axis `axis`.
If `x` is not a tensor or `dim(x, axis) % num_results != 0`, returns `None`.

* `is_defined_in_parent_scope(x: Value) -> Value` is defined on strings
  and returns `true` if `x` is the name of a function defined in the same scope
  as the parent function of the relevant op.

* `is_namespaced_op_name(x: Value) -> Value` is defined on strings and returns
  `true` if `x` is a valid op name, that is it respects the following regular
  expression: `[a-zA-Z][a-zA-Z0-9_]*([.][a-zA-Z0-9_$]+)+`

#### Shape computations

* `axes(x: Value | Placeholder | Type) -> Value` is a shortcut for
`range(rank(x))`.

* `dim(x: Value | Placeholder | Type, axis: Value) -> Value` is a shortcut for
`shape(x)[axis]`.

* `dims(x: Value | Placeholder | Type, axes: List) -> List` is a shortcut for
`list(map(lambda axis: dim(x, axis), axes))`.

* `index_space(x: Value | Placeholder | Type) -> Value` is defined on tensors
and returns `size(x)` indices for the corresponding `TensorType` sorted in
ascending lexicographical order, i.e. `[0, ..., 0]`, `[0, ..., 1]`, ...,
`shape(x) - 1`. If `x` is not a tensor type, a quantized tensor type, or a value
or a placeholder of one of these types, returns `None`.

* `rank(x: Value | Placeholder | Type) -> Value` is a shortcut for
`size(shape(x))`.

* `shape(x: Value | Placeholder | Type) -> Value` is defined in the "Functions
on types" section via `member_name`.

* `size(x: Value | Placeholder | Type) -> Value` is a shortcut for
`reduce(lambda x, y: x * y, shape(x))`.

#### Quantization computations

* `def baseline_element_type(x: Value | Placeholder | Type) -> Type` is a
shortcut for `element_type(baseline_type(x))`.

* `baseline_type` is defined on tensor types and quantized tensor types and
transforms them to a "baseline", i.e. a type with the same shape but with the
quantization parameters of the element type reset to default values.  This is
used as a handy trick to compare both tensor and quantized tensor types
uniformly, which is needed quite often. For quantized types, this enables
comparing types ignoring the quantization parameters, that is, `shape`,
`storage_type`, `expressed_type`, `storage_min`, `storage_max`, and
`quantization_dimension` (for per-axis quantized type) must all match, but
`scales` and `zero points` may differ.

```python
def baseline_type(x: Value | Placeholder | Type) -> Type:
  if type(x) == TensorType:
    return x
  if type(x) == QuantizedTensorType:
    element_type = quantized_tensor_element_type(x)
    baseline_element_type = QuantizedTensorElementType(
      storage_type = storage_type(element_type),
      storage_min = storage_min(element_type),
      storage_max = storage_max(element_type),
      expressed_type = expressed_type(element_type),
      quantization_dimension = quantization_dimension(element_type),
      scales = [constant(1.0, expressed_type(element_type))] * dim(x, quantization_dimension(element_type)),
      zero_points = [constant(0, storage_type(element_type))] * dim(x, quantization_dimension(element_type)))
    return QuantizedTensorType(shape(x), baseline_element_type)
  if type(x) is not Type:
    return baseline_element_type(type(x))
```

* `dequantize` is defined on quantized tensor types and turns them into
floating-point tensor types. This happens via converting quantized elements
which represent integer values of the storage type into corresponding
floating-point values of the expressed type using the zero point and scale
associated with the quantized element type.

```python
def compute_zero_points(quantized_type, result_type):
  if is_per_tensor_quantized(quantized_type):
    return broadcast_in_dim(constant(zero_point(quantized_type), storage_type(quantized_type)), [], result_type)
  if is_per_axis_quantized(quantized_type):
    for i in index_space(result_type):
      d = quantization_dimension(quantized_type)
      zero_points[i] = zero_points(quantized_type)[i[d]]
    return zero_points

def compute_scales(quantized_type, result_type):
  if is_per_tensor_quantized(quantized_type):
    return broadcast_in_dim(constant(scale(quantized_type), expressed_type(quantized_type)), [],
            type(result_type))
  if is_per_axis_quantized(quantized_type):
    for i in index_space(result_type):
      d = quantization_dimension(quantized_type)
      scales[i] = scales(quantized_type)[i[d]]
    return scales

def dequantize(x: Value) -> Value:
  assert is_quantized(x)
  x_storage = bitcast_convert(x, storage_type(x))
  x_storage_sub = x_storage - compute_zero_points(type(x), type(x_storage))
  x_expressed_sub = convert(x_storage_sub, expressed_type(x))
  return x_expressed_sub * compute_scales(type(x), type(x_expressed_sub))
```

* `quantize` is defined on floating-point tensor types and turns them into
quantized tensor types. This happens via converting floating-point values
of the expressed type into corresponding integer values of the storage type
using the zero point and scale associated with the quantized element type.

```python
def quantize(x: Value, result_type: Type) -> Value:
  assert is_float(x) and is_quantized(result_type)
  zero_points = compute_zero_points(result_type, TensorType(shape(x), storage_type(result_type)))
  converted_zero_points = convert(zero_points, expressed_type(result_type))
  converted_min = convert(storage_min(result_type), expressed_type(result_type))
  converted_max = convert(storage_max(result_type), expressed_type(result_type))

  x_scaled = x / compute_scales(result_type, type(x))
  x_scaled_add_zp = x_scaled + converted_zero_points
  x_clamped = clamp(converted_min, x_scaled_add_zp, converted_max)
  x_rounded = round_nearest_even(x_clamped)
  return convert(x_rounded, result_type)
```

* `dequantize_op_quantize` is used to specify element-wise computations on
quantized tensors. It dequantizes, i.e. turns quantized elements into their
expressed types, then performs an operation, and then quantizes, i.e. turns
the results back into their storage types. At the moment, this function only
works for per-tensor quantization. Per-axis quantization is work in progress
([#1574](https://github.com/openxla/stablehlo/issues/1574)).

```python
def dequantize_op_quantize(op, *inputs_and_output_type):
  inputs = inputs_and_output_type[:-1]
  output_type = inputs_and_output_type[-1]

  float_inputs = map(dequantize, inputs)
  float_result = op(*float_inputs)
  return quantize(float_result, output_type)

def dequantize_batch_norm_grad_or_training_quantize(op, *inputs_and_output_types):
  inputs = inputs_and_output_type[:-3]
  float_inputs = map(dequantize, inputs)
  float_results = op(*float_inputs)
  return map(quantize, float_results, inputs_and_output_type[-3:])

def dequantize_compare(lhs, rhs, comparison_direction):
  float_lhs = dequantize(lhs)
  float_rhs = dequantize(rhs)
  return compare(float_lhs, float_rhs, comparison_direction, FLOAT)

def dequantize_select_quantize(pred, on_true, on_false, output_type):
  float_on_true = dequantize(on_true)
  float_on_false = dequantize(on_false)
  float_result = select(pred, float_on_true, float_on_false)
  return quantize(float_result, output_type)
```

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

#### Grid computations

* `cross_partition(replica_groups: Value) -> Value`. See the "cross_replica"
section above.

* `cross_replica(replica_groups: Value) -> Value`. See the "cross_replica"
section above.

* `cross_replica_and_partition(replica_groups: Value) -> Value`. See the
"cross_replica_and_partition" section above.

* `flattened_ids(replica_groups: Value) -> Value`. See the "flattened_ids"
section above.

## Dynamism

StableHLO values can have dynamic dimension sizes, e.g. `tensor<?xi64>`.
However, StableHLO values cannot have a dynamic number of dimensions (unranked
dynamism, e.g. `tensor<*xi64>`). Operands and results are allowed to use dynamic
dimension sizes, even if there are constraints on the sizes. Constraints will be
verified statically if possible, otherwise they are deferred to runtime and
mismatches will result in undefined behavior. See below for examples.

### Shape mismatches for unary elementwise operations

Consider the following toy program:

```mlir
func.func @foo(%arg0: tensor<?xf64>) {
  %0 = stablehlo.abs %arg0 : (tensor<?xf64>) -> tensor<2xf64>
  return
}
```

Such a program is unusual, because it is not common to know the shape of the
result but not the shape of the input. Nonetheless, this is a valid StableHLO
program. It is not possible to statically validate the `abs` operation in this
program, because the exact shape of the operand is unknown. However, the shapes
are certainly compatible, and this can be checked statically: `?` could turn out
to be `2` at runtime, and there would be no issue. However, `?` could
also turn out to be some other integer, in which case the behavior is undefined.

Note that if a dimension size is dynamic in the result, there cannot be
undefined behavior. Indeed, there is no "expected" size, so there cannot be a
mismatch.

### Shape mismatches for binary elementwise operations

Consider the following toy program:

```mlir
func.func @foo(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) {
  %0 = stablehlo.add %arg0, %arg0 : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return
}
```

When it comes to binary elementwise operations, the shapes of the inputs and the
result must agree at runtime. At compile time, static dimensions must be equal,
otherwise they merely need to be compatible.
If *any* dimension is dynamic in the inputs, then there could be undefined
behavior at runtime, because the dynamic size may not match the corresponding
size in the other operand (be it static or dynamic). If all the inputs are
static, then whether the result is dynamic or not does not matter: statically
known dimensions will be checked statically, and dynamic dimensions do not
impose any constraints.

### Shape mismatches for ops that take their output shape as an operand

Consider the following toy program:

```mlir
func.func @foo(%arg0: tensor<2xi32>) {
  %0 = stablehlo.dynamic_iota %arg0, dim = 0 : (tensor<2xi32>) -> tensor<3x4xi64>
  return
}
```

The values in the shape operand at runtime must match the shape of the result,
otherwise the behavior is undefined. That is, at runtime `%arg0` must have a
value of `dense<[3, 4]> : tensor<2xi32>`. If the shape operand is constant, this
can be verified statically. If the result shape is fully dynamic, then there
cannot be a mismatch.
