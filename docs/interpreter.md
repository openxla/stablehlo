# Interpreter Design


## Class Hierarchy

At the moment, StableHLO programs are computations over Tensors (n-dimensional
    arrays), hence most of the input/output data values for an op are Tensors
with the exception of constant
[Attribute](https://mlir.llvm.org/docs/LangRef/#attributes) values of primitive
types (e.g., Int, float etc.).

In the current design, the class `Tensor` implements Tensor data. The underlying
storage class for `Tensor` objects, `detail:Buffer`, stores the type of the
Tensor along with a contiguous byte array representing its data.

Individual elements of a Tensor are represented using `Element` class which uses
`mlir::Attribute` for storage.

`Tensor`
class has the following APIs to interact with it's individual elements:
  - `Element Tensor::get(int64_t index)`: To extract an individual tensor
  element at index `index` as `Element` object.
  - `void Tensor::set(int64_t index, Element element);`: To insert an `Element`
  object `element` into a Tensor at index `index`.

## Working of the interpreter

The entry function to the interpreter is

```C++
SmallVector<Tensor> eval(func::FuncOp func, ArrayRef<Tensor> args);
```

which does the followings:
1. Maps SSA values with `Tensor` values.
2. Invokes `eval` on each of the ops within `func`.

The op-level `eval` as mentioned in (2) is responsible for implementing the
execution semantics of the op. Following is an example for `stablehloadd` op.
In the example, individual elements of the `lhs` and `rhs` Tensors are pairwise
extracted as `Element` objects which are then added. The result of the addition,
          an `Element` object, is stored in the final `result` Tensor.


```C++
Tensor eval(AddOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto i = 0; i < lhs.getNumElements(); ++i) {
    result.set(i, lhs.get(i) + rhs.get(i));
  }
  return result;
}
```

In general, the `Element` class simplifies the execution semantics as specified
in the `eval` function by encapsulating details about how different types are
handled.

## Interpreter used for constant folding
We can use the interpreter mechanism to fold operations with constant
operand/attribute vales.

A high level idea of the implementation would looks like
  - Check if an op is constant-foldable.
  - If yes, invoke the op-level `eval` function.

## Testing the interpreter

The interpreter takes as inputs (A) a
StableHLO program, and (B) data values to be fed to the program, and generates
output data values, which are matched against the user-provided expected data
values.

In the current implementation, we package the inputs (mlir program + input data
    values) and outputs in a
[lit-based]((https://llvm.org/docs/CommandGuide/lit.html)) unit-test as follows:

```c++
// CHECK-LABEL: Evaluated results of function: add_op_test_ui4
// CHECK-NEXT:  tensor<2xui4>
// CHECK-NEXT:    15 : ui4
// CHECK-NEXT:    5 : ui4

func.func @add_op_test_ui4() -> tensor<2xui4> {
  %0 = stablehlo.constant dense<[0, 2]> : tensor<2xui4>
  %1 = stablehlo.constant dense<[15, 3]> : tensor<2xui4>
  %2 = stablehlo.add %0, %1 : tensor<2xui4>
  func.return %2 : tensor<2xui4>
}
```

A test utility `stablehlo-interpreter-runner`
([code](https://github.com/openxla/stablehlo/tree/main/stablehlo/reference/tests/StablehloInterpreterRunner.cpp))
is responsible for parsing the program and interpret each function and return
the resulting tensor(s) to be matched against the output tensor provided in
[FileCheck directives](https://llvm.org/docs/CommandGuide/FileCheck.html). We
have a dedicated test-suite, consisting of several unit-tests, for each
StableHLO Op. The unit-suites can be found
[here](https://github.com/openxla/stablehlo/tree/main/stablehlo/reference/tests).

