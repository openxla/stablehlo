# [RFC] Add async support to the StableHLO specification

Status: Under Review<br/>
Initial version: 09/17/2024<br/>
Last updated: 09/17/2024<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/2551)


## Motivation

Today, stableHLO ops are designed to be executed sequentially, and any async dispatch or scheduling is left to the compiler to define.

However, getting XLA to generate optimized schedules has proven to be very challenging.
Users have found that this leaves a lot of performance on the table, and have vocalized a desire to have more control over the scheduling.

There is an [excellent write up](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)
from Yifu Wang that goes into detail the performance benefits of async tensor parallelism. 

There is already existing async infrastructure in XLA that we use to create collective matmuls, so the main goal 
is to expose this in stableHLO and have it accessible in JAX's `shard_map`.


## Proposed Specification changes

### Types


```ebnf
AsyncType ::= 'async' '<' ValueType '>'
```
*Async Types* represents tensor values that must be awaited on before using the underlying values. Async operations 
allow multiple operations to be running at once as described in the Async Execution section.

Add `AsyncType` to `NonValueType`

```ebnf
NonValueType ::= TensorElementType | QuantizedTensorElementType | FunctionType | StringType | AsyncType
```

### Ops

### async_start

#### Semantics

Produces the output from executing the `body` function, but runs all operations on a stream separate from the main compute stream.

The output of an `async_start` computation must first be processed by an `async_done` operation.

#### Inputs

| Label | Name      | Type                                                    | Constraints |
|-------|-----------|---------------------------------------------------------|-------------|
| (I1)  | `operand` | variadic number of tensors, quantized tensors or tokens | (C1)        |
| (I2)  | `body`    | function                                                | (C1)        |

#### Outputs

| Name      | Type                               | Constraints |
|-----------|------------------------------------|-------------|
| `results` | variadic number of async values    | (C1)        |

#### Constraints

* (C1) `body` has type `(T0, ..., TN-1) -> (R0, ..., RM-1)`, where
       `type(operand[i]) == Ti` and `type(results[i]) == async<Ri>`

#### Examples

```mlir
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

### async_done

#### Semantics

Waits for the values created by an `async_start` operation to be finalized. All tensors given to `async_done` must has type `async<T>`.

#### Inputs

| Label | Name      | Type                               | Constraints |
|-------|-----------|------------------------------------|-------------|
| (I1)  | `operand` | variadic number of async values    | (C1)        |

#### Outputs

| Name      | Type                                                    | Constraints |
|-----------|---------------------------------------------------------|-------------|
| `results` | variadic number of tensors, quantized tensors or tokens | (C1)        |

#### Constraints

* (C1) `type(operand) == (async<T0>, ..., async<TN>)` and `type(result) == (T0, ..., TN)`

#### Examples

```mlir
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


## Execution

### Async Execution

Stable HLO programs have dataflow semantics, and each backend is free to dispatch execution in any 
pattern as long as it respects data dependencies. But this usually means that only one operation is run at a time. 
However, the ops `async_start` and `async_done`, along with barriers or `token` management would allow you to define data dependencies that can force one 
operation to start before another finishes. This could allow you to better utilize your hardware or to define your own communication schedule.
Async operations are an advanced tool that should only be used when you know what you are doing.
