# [RFC] Standardize Tuple-collective (AllGatherOp, AllReduceOp, AllToAllOp) ops

Status: Review<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/12/2024<br/>
Discussion thread: [GitHub](add PR Link)

## Motivation

Several features have been added to MHLO in the past year, which frameworks want
to leverage and members of the community have made requests for them as well.
This includes: support of variadic operands/results for tuple-collective
(`AllGatherOp`,`AllReduceOp`, `AllToAllOp`) ops. We propose adding these
feature to the StableHLO spec so they can be used by the community.

StableHLO tuple-collective ops support is limited to **single-operand** and **single-result**.
[MHLO ops](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td)
supports
**multi-operand** and **multi-result** which is in sync with multi-operand and
multi-result XLA semantics and
horizontal scaling
[`all_reduce`](https://openxla.org/xla/operation_semantics#allreduce)
[`all_gather`](https://openxla.org/xla/operation_semantics#allgather) and
[`all_to_all`](https://openxla.org/xla/operation_semantics#alltoall). `all_reduce`
support is requested
in [#1370](https://github.com/openxla/stablehlo/issues/1370) is relied on by
PyTorch/XLA today via XlaBuilder ([ref](https://github.com/pytorch/xla/blob/1bbe333ad137ace6b8134db640c0b24c8c428db6/torch_xla/csrc/cross_replica_reduces.cpp#L156)).
`all_to_all` support is requested in
[#574](https://github.com/openxla/stablehlo/issues/574) and identified as a feature
gap.

## Proposed Specification Changes

View spec.md changes in this PR to view the diff vs original spec.

To view rich text of the spec, see:
[AllGatherOp](https://github.com/openxla/stablehlo/blob/f8d6756c70dc5301d5be88d1ca378d1429943e0c/docs/spec.md#all_gather)
[AllReduceOp](https://github.com/openxla/stablehlo/blob/f8d6756c70dc5301d5be88d1ca378d1429943e0c/docs/spec.md#all_reduce)
[AllToAllOp](https://github.com/openxla/stablehlo/blob/f8d6756c70dc5301d5be88d1ca378d1429943e0c/docs/spec.md#all_to_all)
