# [RFC] Add collective_broadcast to the StableHLO specification

Status: Approved<br/>
Initial version: 10/17/20223<br/>
Last updated: 11/1/2023<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/1809)

## Motivation

StableHLO currently has [five collective communication primitives](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective-ops):
`collective_permute`, `all_gather`, `all_to_all`, `all_reduce`, and
`reduce_scatter`. However, one of the major collective communication
primitives, `broadcast`, is missing from this list. This primitive allows for a
one-to-many replication of a tensor to many devices efficiently. `broadcast` is
a primitive in [MPI](https://www.open-mpi.org/doc/v4.1/man3/MPI_Bcast.3.php),
[NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclBroadcast),
and [PyTorch](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast).
From here on out, we will refer to this operation as `collective_broadcast` for
reasons discussed later.

While it technically would be possible to replicate a broadcast with a
conditional mask and a `psum`, that reduces to an `all_reduce` communication
primitive, which is significantly more expensive than a simple
`collective_broadcast`. Additionally, when dealing with network-switch
environments, the explicit use of `collective_broadcast` allows the switch to
greatly optimize it's throughput when replicating to many targets
simultaneously. However, XLA currently has no ability to lower directly to a
mesh's `collective_broadcast` primitive, so a lot of that optimization is left
on the table.

Additionally, a new compiler pass that detects usage of the old `psum` hack and
replaces it with a `collective_broadcast` could be implemented only once and
forever be supported by all hardware, future and current. This could have
positive knock-on effects for users who don't even realize they're using it!

`collective_broadcast` can be used to quickly replicate a tensor across an
entire mesh, and would use less communication resources as compared to
`all_gather` or `psum`. `collective_broadcast` is also the base primitive used
in the [SUMMA](https://www.netlib.org/lapack/lawnspdf/lawn96.pdf) distributed
GEMM algorithm. As AI computing grows larger, there likely will grow a need for
these 2D distributed GEMM algorithms. Adding support for one of the needed
primitives could help advance research in these areas.

## Alternatives considered

Instead of adding `collective_broadcast` as a primitive, we considered
loosening the restriction of `collective_permute` to allow a one-to-many
communication schedule instead of the current restriction of a one-to-one
schedule. Downstream compilers would then be responsible for detecting this and
calling their own `collective_broadcast` primitive. However, loosening this
restriction makes defining the transposition rule for `collective_permute`
significantly more complicated. Questions of how to calculate that and do it
efficiently given any communication configuration and do so in SPMD became
difficult. However, the transposition rule for `collective_broadcast` is just
`psum` with a source-device one-hot masking. This simplicity plus the broad
usage of `collective_broadcast` in the wider ecosystem made us choose to
ultimately add the new primitive instead.

## Why call it collective_broadcast and not just broadcast?

Unfortunately, the op name `broadcast` is already taken by [an op in XLA proper](https://www.tensorflow.org/xla/operation_semantics#broadcast),
so we can't have the two names clash. `collective_broadcast` was the preferred
alternative.

## Proposed Specification

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

* `operand@process_groups[i, 0]` if there exists an `i` such that
  the process is in `process_groups[i]`.
* `broadcast_in_dim(constant(0, element_type(result)), [], type(result))`
  otherwise.

#### Inputs

| Label | Name                    | Type                                                             | Constraints |
|-------|-------------------------|------------------------------------------------------------------|-------------|
| (I1)  | `operand`               | tensor                                                           | (C3)        |
| (I2)  | `replica_groups`        | variadic number of 1-dimensional tensor constants of type `si64` | (C1), (C2)  |
| (I3)  | `channel_id`            | constant of type `si64`                                          |             |

#### Outputs

| Name     | Type   | Constraints |
|----------|--------|-------------|
| `result` | tensor | (C3)        |

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
