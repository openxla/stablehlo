# Interpreter Design

## Data Model

[StableHLO programs](spec.md#programs) are computations over tensors
(n-dimensional arrays), which, in the current model, are implemented using class
`Tensor`. The underlying storage class for a `Tensor` object, `detail::Buffer`,
stores the `mlir::ShapedType` of the tensor along with a
`mlir::HeapAsmResourceBlob` object representing a mutable blob of tensor
data laid out as contiguous byte array in
[major-to-minor order](https://www.tensorflow.org/xla/shapes).
`detail::Buffer` objects are reference-counted to simplify memory management.

Individual elements of a tensor are represented using `Element` class which uses
discriminated union holding one of `APInt`, `APFloat` or `pair<APFloat,APFloat>`
for storage. The last one is used for storing elements with complex types.

`Tensor` class has the following APIs to interact with its individual elements:

- `Element Tensor::get(llvm::ArrayRef<int64_t> index)`: To extract an
  individual tensor element at multi-dimensional index `index` as `Element`
  object.
- `void Tensor::set(llvm::ArrayRef<int64_t> index, Element element);`:
  To update an `Element` object `element` into a tensor at multi-dimensional
  index `index`.

## Working of the interpreter

The entry function to the interpreter is

```C++
SmallVector<Tensor> eval(func::FuncOp func, ArrayRef<Tensor> args);
```

which does the following:

1. Tracks the SSA arguments of `func` and their associated runtime `Tensor`
   values, provided in `args`, using a symbol table map, M.
2. Foreach op within `func` in their SSACFG order:
   - Invokes `eval` on op. For each SSA operand of the op, extract its
     runtime value from M to be provided as argument to the `eval` invocation.
   - Tracks the SSA result(s) of the op and the evaluated value in M.

The op-level `eval` as mentioned in (2) is responsible for implementing the
execution semantics of the op. Following is an example for `stablehlo::AddOp`.
In the example, individual elements of the `lhs` and `rhs` tensors are pairwise
extracted as `Element` objects which are then added. The result of the addition,
an `Element` object, is stored in the final `result` tensor.

```C++
Tensor eval(AddOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());

  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) + rhs.get(*it));

  return result;
}
```

Overall, the design of the interpreter is optimized for readability of
implementations of `eval` functions for individual ops because it's meant to
serve as a reference implementation for StableHLO. For example, instead of
defining `eval` as a template function and parameterizing it with element types,
we encapsulate details about how different element types are handled in
`Element::operator+` etc, simplifying the implementation of `eval`.

## Using interpreter for constant folding

We can use the interpreter mechanism to fold operations with constant operand
values. The following code snippet demonstrates an idea of the implementation
for folding `stablehlo::AddOp` with floating-point typed operands:

```C++
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto attrs = adaptor.getOperands();
  DenseElementsAttr lhsData = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhsData = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhsData || !rhsData) return {};

  auto lhs = Tensor(lhsData);
  auto rhs = Tensor(rhsData);
  auto result = eval(*this, lhs, rhs);

  SmallVector<APFloat> values;
  for (auto i = 0; i < result.getNumElements(); ++i) {
    Element element = result.get(i);
    values.push_back(element.getValue().cast<FloatAttr>().getValue());
  }

  return DenseElementsAttr::get(result.getType(), values);
}
```

At the moment, we aren't actively working on integrating the interpreter into
constant folding because we aren't planning to implement folder for StableHLO.
However, in the future, we are planning to leverage the interpreter for constant
folding in MHLO, at which point we'll improve ergonomics of the code snippet
above (e.g. we could have a helper function which packs constant operands into
`Tensor` objects and unpacks `Tensor` results into `OpFoldResult`).

## Testing the StableHLO interpreter

The interpreter takes as inputs (A) a StableHLO program, and (B) data values to
be fed to the program, and generates output data values, which are matched
against the user-provided expected data values. The data values (B) are
hard-coded in the program itself using `stablehlo.constant` operations. The
interpreter evaluates the input program. The output(s) of the op under test
is checked via checks (e.g. `check.expect_eq`, `check.expect_almost_eq`), as
shown below. `check.expect_eq` and `check.expect_eq_const` check for bitwise
equality for any supported type and `check.expect_almost_eq` and
`check.expect_almost_eq_const` check for near equality within a tolerance,
explained in testing guideline (G6), for floating point and complex types.

```C++
// CHECK-LABEL: Evaluated results of function: add_op_test_ui4
func.func @add_op_test_ui4() {
  %0 = stablehlo.constant dense<[0, 2]> : tensor<2xui4>
  %1 = stablehlo.constant dense<[15, 3]> : tensor<2xui4>
  %2 = stablehlo.add %0, %1 : tensor<2xui4>
  check.expect_eq_const %2, [15, 5] : tensor<2xui4>
  func.return
}
```

A test utility `stablehlo-translate --interpret`
([code](https://github.com/openxla/stablehlo/tree/main/stablehlo/tools/StablehloTranslateMain.cpp))
is responsible for parsing the program, interpreting each function including the
operations constituting the function. We have a dedicated test-suite, consisting
of several tests exercising various runtime behaviors, for each StableHLO Op.
The tests can be found [here](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/)
(e.g. interpret\_\*.mlir).

### Testing guidelines

**(G1) Do we need to test for all the supported types for every op?**

We can use a combination of following rules to decide it:

1. While implementing an op, if there exists code in the corresponding `eval`
   function to handle a particular type, then it is imperative to have test(s)
   to cover for that type. As an example, for `add` op, there is exclusive code
   to handle integer, boolean, floating-point, and complex types, and hence we
   need one test for each category of types.

2. If a set of types is handled uniformly in the corresponding `eval` function,
   then a single test for all those types should be sufficient. As an example,
   for `add` op, all the variants of integer types (`si4`, `u4`, `si8`, `u8` and
   so on) are handled alike using `llvm::APInt` APIs, and hence we can skip
   adding tests for each of those variants, and instead, add a single
   representative test. To avoid ambiguity in selecting the representative, we
   should use the following guidelines:

     - If all the types, handled uniformly, have the same primitive type
       (i.e., if all are integer, or floating-point, or complex types), then
       choose the one with maximum bit-width.
     - If all the types, handled uniformly, have a mix of primitive types, then
       choose the one with the following primitive type, in decreasing order of
       preference: integer, floating-point, boolean, complex.

**(G2) How do we decide on the number of tests needed to cover an op's
behavior?**

The goal is to comprehensively cover the logic of the interpreter for the op
(i.e. all corner cases of the implementation) with a minimal number of tests.
Minimizing the number of tests is important for maintainability. The fewer tests
we have, the easier it is to review them and to make sure that they
comprehensively cover the op. As a result, we expect that most of the simpler
ops will end up having just one test. If due to some good reason comprehensive
coverage is impractical, then it is fine to stop at >= 90%. This will be decided
on case-by-case basis during pull request review.

**(G3) How about adding tests dedicated for testing the interpreter
infrastructure?**

The interpreter infrastructure is mostly straightforward and can be added to
our trust base. The only non-trivial part is how various types are packed into
and unpacked from the underlying interpreter storage. As discussed in (G1), we
will be testing only those types of an op which are handled differently. With
that it is possible that the packing/un-packing code, corresponding to different
variants of integer/floating-point types, might not get fully covered during
testing. To ensure that we can choose an op, like `constant`, which supports all
the StableHLO element types and write exhaustive tests.

**(G4) If the implementation of an op depends other ops, should be write tests
for the latter?**

No. For example, the implementation of `batch_norm_grad` can be based on
`divide`, `subtract`, `multiply` and others, we should avoid testing the latter
ops while testing the former.

**(G5) Should we write tests to exercise the implementation-defined / undefined
behaviors?**

We should not write tests which exercise the implementation defined or
undefined behaviors of the op. Tests exercising implementation defined behaviors
demonstrate a local behavior of the interpreter which should not be
generalized. Tests exercising undefined behavior do not contribute towards
the understanding of the op's behavior.

**(G6) While writing tests for floating-point type, to what precision the
expected result need to be specified in checks?**

For elementary operations (addition, subtraction, multiplication, division, and
square), an implementation following IEEE specification is expected to provide a
rounded result within 0.5 ULP of the mathematically exact result. That said, we
can safely imagine the expected result coming out of these operations to be at
most 1 ULP apart. However, this may not work for transcendental functions
(`sine`, `cosine`, etc.) for which the precision guarantees are
implementation-defined ([rationale](https://github.com/openxla/stablehlo/issues/96)).

The current implementation uses a "one-size-fits-all" tolerance value of 0.0001.
The following example demonstrates the above tolerance in action.

```mlir
func.func @check_tolerance() {
  %0 = stablehlo.constant dense<0.2> : tensor<f32>

  // The following check succeeds as %0 is almost equal to the provided
  // constant modulo the tolerance, mentioned above.
  check.expect_almost_eq_const %0, dense<0.19999> : tensor<f32>

  // The following check fails as %0 is not bitwise equal to the provided
  // constant.
  check.expect_eq_const %0, dense<0.19999> : tensor<f32>

  func.return
}
```

This is just the first step in testing numerical accuracy of StableHLO ops. At
the moment, this is an underspecced area of the StableHLO spec, and there is
ongoing work to figure it out [#1156](https://github.com/openxla/stablehlo/issues/1156)
based on our experience with using StableHLO in practice and on feedback from
stakeholders. As this works proceeds, we will update this infrastructure
accordingly.

**(G7) Anything about the coding-style of the tests?**

1. Make sure to use the actual name of the inputs/outputs instead of defaulting
   to SSA values (e.g. %0, %1, etc.)
1. Make sure the tests use pretty-printed format, if it exists.

**(G8) Should we include the example already provided in the spec?**
Yes (for completeness of testing).
