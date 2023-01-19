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
OpFoldResult AddOp::fold(ArrayRef<Attribute> attrs) {
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

## Testing the interpreter

The interpreter takes as inputs (A) a StableHLO program, and (B) data values to
be fed to the program, and generates output data values, which are matched
against the user-provided expected data values.

In the current implementation, we package the inputs (MLIR program + input data
values) and outputs in a
[lit-based](https://llvm.org/docs/CommandGuide/lit.html) test as follows:

```C++
// CHECK-LABEL: Evaluated results of function: add_op_test_ui4
func.func @add_op_test_ui4() -> tensor<2xui4> {
  %0 = stablehlo.constant dense<[0, 2]> : tensor<2xui4>
  %1 = stablehlo.constant dense<[15, 3]> : tensor<2xui4>
  %2 = stablehlo.add %0, %1 : tensor<2xui4>
  func.return %2 : tensor<2xui4>
  // CHECK-NEXT:  tensor<2xui4>
  // CHECK-NEXT:    15 : ui4
  // CHECK-NEXT:    5 : ui4
}
```

A test utility `stablehlo-interpreter`
([code](https://github.com/openxla/stablehlo/tree/main/stablehlo/tools/StablehloInterpreterMain.cpp))
is responsible for parsing the program, interpreting each function, and
returning the resulting tensor(s) to be matched against the output tensor
provided in [FileCheck
directives](https://llvm.org/docs/CommandGuide/FileCheck.html). We have a
dedicated test-suite, consisting of several tests exercising various runtime
behaviors, for each StableHLO Op. The tests can be found
[here](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/) (e.g.
interpret\_\*.mlir).

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

     - If all the types, handled uniformaly, have the same primitive type
       (i.e., if all are integer, or floating-point, or complex types), then
       choose the one with maximum bit-width.
     - If all the types, handled uniformaly, have a mix of primitive types, then
       choose the one with the following primitive type, in decreasing order of
       preference: integer, floating-point, boolean, complex.

**(G2) How about adding tests dedicated for testing the interpreter
infrastructure?**

The interpreter infrastructure is mostly straightforward and can be added to
our trust base. The only non-trivial part is how various types are packed into
and unpacked from the underlying interpreter storage. As discussed in (G1), we
will be testing only those types of an op which are handled differently. With
that it is possible that the packing/un-packing code, corresponding to different
variants of integer/floating-point types, might not get fully covered during
testing. To ensure that we can choose an op, like `constant`, which supports all
the StableHLO element types and write exhaustive tests.

**(G3) If the implementation of an op depends other ops, should be write tests
for the latter?**

No. For example, the implementation of `batch_norm_grad` can be based on
`divide`, `subtract`, `multiply` and others, we should avoid testing the latter
ops while testing the former.

**(G4) Should we write tests to exercise the implementation-defined / undefined
behaviors?**

We should not write tests which exercise the implementation defined or
undefined behaviors of the op. Tests exercising implementation defined behaviors
demonstrate a local behavior of the interpreter which should not be
generalized. Tests exercising undefined behavior do not contribute towards
the understanding of the op's behavior.

**(G5) While writing tests for floating-point type, to what precision the
results need to be specified in llvm lit checks?**

The current lit-based interpreter testing fails if the result is computed with a
different precision than what is mentioned in the lit CHECK directives. As a
quick-fix, we allow the lit checks to measure the accuracy up to an arbitrary
places after the decimal point. But the solution is far from ideal. We plan to
resolve it using [ticket](https://github.com/openxla/stablehlo/issues/268).
