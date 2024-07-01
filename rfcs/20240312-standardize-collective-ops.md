# [RFC] Standardize collective ops to support variadic operand/result

Status: Approved<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/15/2024<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/2099)

## Motivation

Several features have been added to MHLO in the past year, which frameworks want
to leverage and members of the community have made requests for them as well.
This includes: feature to support variadic operands/results for collective
(`AllGatherOp`,`AllReduceOp`, `AllToAllOp`) ops.

We propose adding this feature to the StableHLO spec so they can be used by the community.
StableHLO collective ops support is currently limited to **single-operand** and **single-result**.
[MHLO collective ops](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td)
support
**multi-operand** and **multi-result** which is in sync with multi-operand and
multi-result XLA semantics
([`all_reduce`](https://openxla.org/xla/operation_semantics#allreduce),
[`all_gather`](https://openxla.org/xla/operation_semantics#allgather) and
[`all_to_all`](https://openxla.org/xla/operation_semantics#alltoall)) and
horizontal scaling. `all_reduce`
support is requested
in [#1370](https://github.com/openxla/stablehlo/issues/1370) and is relied on by
PyTorch/XLA today via XlaBuilder ([ref](https://github.com/pytorch/xla/blob/1bbe333ad137ace6b8134db640c0b24c8c428db6/torch_xla/csrc/cross_replica_reduces.cpp#L156)).
`all_to_all` support is requested in
[#574](https://github.com/openxla/stablehlo/issues/574) and identified as a feature
gap.

## Proposed Specification Changes

Please refer spec.md changes in this PR to view the diff vs original spec.
