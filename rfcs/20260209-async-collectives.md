# [RFC] Async Collectives

Status: In Review<br/>
Initial version: 02/09/2026<br/>
Last updated: 02/09/2026<br/>
Discussion thread: N/A

## Motivation

StableHLO programs can do two things: perform local computation (e.g., matrix
multiplication) and exchange data via collectives (e.g., an all-reduce). To get
high performance, it is crucial that these two things are overlapped. Local
computation should be executed while collectives are running in the background,
whenever possible.

Today, StableHLO doesn't implement any kind of communication-compute overlap,
though [XLA does][async_hlo]. The six StableHLO collective
operations---all_gather, all_reduce, all_to_all, collective_broadcast,
collective_permute, and reduce_scatter---are lowered to HLO equivalents.
Internally, [the XLA compiler splits these operations into asynchronous
start/done pairs][async_collective_creator]. For example, an `all-reduce`
operation because a pair of an `all-reduce-start` and `all-reduce-done`. Then,
the XLA scheduler---the component responsible for picking the order in which to
run ops---can schedule local computation between a start/done pair.

The XLA scheduler is not perfect. Sometimes, it picks bad schedules. That's why
we want to allow JAX programmers to manually specify (or at least influence) how
their programs are scheduled. This RFC proposes adding asynchronous collectives
to StableHLO, which is one step towards this goal.

By exposing async collectives in StableHLO (and also in JAX and other
higher-level frameworks), a programmer can write code like the following:

```python
future = all_reduce_start(...)
perform_local_computation(...)
all_reduce_done(future)
```

## Overview

This RFC introduces asynchronous versions of the six existing StableHLO
collective ops. For example, we introduce `all_gather_start` and
`all_gather_done` ops to go along with the existing `all_gather` op. We also
introduce a new future type (e.g., `future<tensor<2xf32>>`) to represent the
output to a start operation.

## Proposed Type Changes

We introduce a new future type as follows.

```ebnf
ValueType ::= TensorType | QuantizedTensorType | TokenType | TupleType | BufferType | FutureType
FutureType ::= 'future' '<' FutureValueType '>'
FutureValueType ::= TensorType | QuantizedTensorType
```

## Proposed Op Changes

We introduce six new **start ops**:

- `all_gather_start`
- `all_reduce_start`
- `all_to_all_start`
- `collective_broadcast_start`
- `collective_permute_start`
- `reduce_scatter_start`

These ops are identical to their non-asynchronous counterparts. They take the
same arguments and have the same constraints. The only difference is that they
return futures. Here's an example:

```
%future = "stablehlo.collective_permute_start"(%operand) {
  source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x2xi64>) -> future<tensor<2x2xi64>>
```

We also introduce six **done ops**.

- `all_gather_done`
- `all_reduce_done`
- `all_to_all _done`
- `collective_broadcast_done`
- `collective_permute _done`
- `reduce_scatter_done`

A done op takes a `future<T>` as an argument and returns a `T`. Continuing the
example above:

```
%result = "stablehlo.collective_permute_done"(%future) : (future<tensor<2x2xi64>>) -> tensor<2x2xi64>
```

Start and done ops must be matched in the obvious way. It is an error to pass
the output of an `all_gather_start`, for example, to `reduce_scatter_done`.

## Alternatives

### Generic Async Ops

https://github.com/openxla/stablehlo/pull/2551 is a StableHLO RFC that proposes
adding generic `async_start` and `async_done` ops that can be used to call any
function asynchronously. Here's an example from the RFC that performs an
asynchronous add:

```
// %init_i: 2
// %init_sum: 3
%future = "stablehlo.async_start"(
    %init_i as %arg0: tensor<i64>,
    %init_sum as %arg1: tensor<i64>)
{
    %new_sum = stablehlo.add %arg1, %arg0 : tensor<i64>
    stablehlo.return %new_sum : tensor<i64>
} : (tensor<i64>, tensor<i64>) -> async<tensor<i64>>

%result = "stablehlo.async_done"(%future): async<tensor<i64>> -> tensor<i64>
// %result: 5
```

This RFC proposes something much simpler yet less powerful. In the future, we
could migrate to generic async ops.

### Tensors Instead of Futures

Start ops could return regular tensors instead of futures. The value of these
tensors, however, would be indeterminate. The tensors should not be used in any
way besides as arguments to done ops. Here's an example:

```
%indeterminate = "stablehlo.collective_permute_start"(%operand) {
  source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x2xi64>) -> tensor<2x2xi64>
%result = "stablehlo.collective_permute_done"(%indeterminate) : (tensor<2x2xi64>) -> tensor<2x2xi64>
```

This approach mirrors how HLO represents asynchronous ops. It also avoids
introducing a new future type. However, it is less type-safe.

### Collective in Types

This RFC has every collective return the same future type. Thus, the following
code is well-typed but erroneous.

```
%future = "stablehlo.collective_permute_start"(%operand) {
  source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<2x2xi64>) -> future<tensor<2x2xi64>>
%result = "stablehlo.all_reduce_done"(%future) : (future<tensor<2x2xi64>>) -> tensor<2x2xi64>
```

We could instead introduce a separate future type for every collective. For
example, `collective_permute_start` could return a
`collective_permute_future<...>`, and `collective_permute_done` could take a
`collective_permute_future<...>` as an argument.

This would introduce more type safety.

[async_collective_creator]: https://github.com/openxla/xla/blob/391c1c5fdadde89ee81886495d32dc32f9238af1/xla/hlo/transforms/collectives/async_collective_creator.h#L38
[async_hlo]: https://openxla.org/xla/async_ops
