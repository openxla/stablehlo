# [RFC] Add source-target pairs to send/recv ops

Status: In Review<br/>
Initial version: 04/21/2025<br/>
Last updated: 04/21/2025<br/>
Discussion thread: N/A

## Overview

This RFC proposes adding a new attribute `source_target_pairs` to `send` and
`recv` ops. `source_target_pairs` allows users to specify peer-to-peer
communication patterns using global device IDs (zero-indexed integers).
Currently this feature is only available on GPUs.

## Background

SPMD-based pipeline parallelism relies on optimizations in XLA to pipeline
send/recv operations in such a way that compute and communication are
overlapped. The user expresses this through collective permutes and relies on
XLA to decompose these into send/recv operations, which are then pipelined
separately, allowing for the staggering that is unique to pipeline parallelism.
The limitation of this approach is that it encapsulates the latency hiding
mechanism in the compiler and allows for little control by the user. When this
mechanism fails, the user has little choice but to debug XLA itself. This RFC is
proposed in conjunction with exposing send/recv operations through the JAX
`shard_map` API.

## Proposed Specification

### send

#### Semantics

Sends `inputs` to a channel `channel_id` and produces a `result` token.

If `is_host_transfer` is `true`, then the operation transfers data to the
host. Otherwise, it transfers data to another device based on the values of
`source_target_pairs`. This flag duplicates the information provided in
`channel_type`, so in the future we are planning to only keep one of them
([#666](https://github.com/openxla/stablehlo/issues/666)).

#### Inputs

| Label | Name                  | Type                                            | Constraints |
|-------|-----------------------|-------------------------------------------------|-------------|
| (I1)  | `inputs`              | variadic number of tensors or quantized tensors |             |
| (I2)  | `token`               | `token`                                         |             |
| (I3)  | `source_target_pairs` | 2-dimensional tensor constant of type `si64`    | (C1-C4)     |
| (I4)  | `channel_id`          | constant of type `si64`                         |             |
| (I5)  | `channel_type`        | enum of `DEVICE_TO_DEVICE` and `DEVICE_TO_HOST` | (C5)        |
| (I6)  | `is_host_transfer`    | constant of type `i1`                           | (C5)        |

#### Outputs

| Name     | Type    |
|----------|---------|
| `result` | `token` |

#### Constraints

* (C1) `dim(source_target_pairs, 1) = 2`.
* (C2) `is_unique(source_target_pairs[:, 0])`.
* (C3) `is_unique(source_target_pairs[:, 1])`.
* (C4) `0 <= source_target_pairs < N`, where `N` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_partitions` if `cross_partition` is used.
* (C5) `channel_type` is defined as:
  * `DEVICE_TO_HOST` if `is_host_transfer = true`,
  * `DEVICE_TO_DEVICE` otherwise.

#### Examples

```mlir
%result = "stablehlo.send"(%operand, %token) {
  source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
  is_host_transfer = true
} : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
```

### recv

#### Semantics

Receives data from a channel with `channel_id` and produces `results`.

If `is_host_transfer` is `true`, then the operation transfers data from the
host. Otherwise, it transfers data from another device based on the values of
`source_target_pairs`. This flag duplicates the information provided in
`channel_type`, so in the future we are planning to only keep one of them
([#666](https://github.com/openxla/stablehlo/issues/666)).

`results` consist of payload values which come first and a token which comes
last. In the future, we are planning to split the payload and the token into two
separate outputs to improve clarity
([#670](https://github.com/openxla/stablehlo/issues/670)).

#### Inputs

| Label | Name                  | Type                                            | Constraints |
|-------|-----------------------|-------------------------------------------------|-------------|
| (I1)  | `token`               | `token`                                         |             |
| (I2)  | `source_target_pairs` | 2-dimensional tensor constant of type `si64`    | (C1-C4)     |
| (I3)  | `channel_id`          | constant of type `si64`                         |             |
| (I4)  | `channel_type`        | enum of `DEVICE_TO_DEVICE` and `DEVICE_TO_HOST` | (C5)        |
| (I5)  | `is_host_transfer`    | constant of type `i1`                           | (C5)        |

#### Outputs

| Name      | Type                                                    | Constraints |
|-----------|---------------------------------------------------------|-------------|
| `results` | variadic number of tensors, quantized tensors or tokens | (C2-C4)     |

#### Constraints

* (C1) `dim(source_target_pairs, 1) = 2`.
* (C2) `is_unique(source_target_pairs[:, 0])`.
* (C3) `is_unique(source_target_pairs[:, 1])`.
* (C4) `0 <= source_target_pairs < N`, where `N` is defined as:
  * `num_replicas` if `cross_replica` is used.
  * `num_partitions` if `cross_partition` is used.
* (C5) `channel_type` is defined as:
  * `DEVICE_TO_HOST` if `is_host_transfer = true`,
  * `DEVICE_TO_DEVICE` otherwise.

#### Examples

```mlir
%results0, %results1 = "stablehlo.recv"(%token) {
  source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
  is_host_transfer = false
} : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
```
