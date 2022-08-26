# Interpreter Design


## Class Hierarchy

Stablehlo programs are computations over Tensors (n-dimensional arrays), hence
most of the input/output data values for an op are Tensors with the exception of
constant
[Attribute](https://docs.google.com/document/d/1GAAK_yHk7q2cZdJ3zBhsiNcrqzfpyjwQ-oWLD7q__Ss/edit#heading=h.5nem09i7p9va)
values of primitive types (e.g., Int, float etc.).

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

The op-level `eval` as  mentioned in (2) is responsible for implementing the
execution semantics of the op. Following is an example for Stablehlo::AddOp.  In
the example, individual elements of the `lhs` and `rhs` Tensors are pairwise
extracted as `Element` objects which are then added. The result of the addition,
          an `Element`  object, is stored in the  final `result` Tensor.


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

This is work-in-progress.
