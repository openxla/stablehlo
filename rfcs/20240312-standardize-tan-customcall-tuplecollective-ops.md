# [RFC] Standardize TanOp, CustomCallOp with typed FFI, and tuple-collectives ops

Status: Draft<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/12/2024<br/>
Discussion thread: [GitHub](add PR Link)

## Motivation

Several features have been added to MHLO in the past year, which frameworks want
to leverage and members of the community have made requests for them as well.
This includes: `TanOp`, `CustomCallOp` with typed FFI (dictionary), and
tuple-collective ops (`AllGatherOp`, `AllReduceOp`, `AllToAllOp` with variadic
operands/results). Some of these features are being used today with various
workarounds -- unregistered attributes, serializing dictionary as string,
custom_calls. None of these approaches are stable, so we propose adding these
ops/features to the StableHLO spec so they can be used safely by the community.

### TanOp

Frameworks and Compilers both want `tan` op.
Jax has [`jnp.tan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tan.html),
PyTorch has [`torch.tan`](https://pytorch.org/docs/stable/generated/torch.tan.html).
On Compilers side, XLA has [`mhlo.tan`](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L633).
StableHLO doesn't support `tan` op, but there is an open ticket which requests
adding this feature in
[#1358](https://github.com/openxla/stablehlo/issues/1358)

### CustomCallOp with typed FFI

StableHLO `custom_call` op supporting `backend_config` dictionary will help to
unify metadata under single `mlir::DictionaryAttr`. This provides more stable
serialization of `custom_call` metadata, a feature that is desired by frameworks
and compilers. There are several occurrences of users working around this gap in
`custom_call` today, either as an unregistered attribute
([example](https://github.com/google/jax/blob/1ed27ecebb92e916b45601e3a107971170a4592b/jaxlib/hlo_helpers.py#L191)),
or a serialized dictionary string. Standardizing this feature to StableHLO will
benefit the entire ecosystem. We propose to support the same as what
[MHLO custom_call op](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L2483)
already supporting. Open tickets for this request: [#637](https://github.com/openxla/stablehlo/issues/637),
[#741](https://github.com/openxla/stablehlo/issues/741)

### Tuple-collectives (AllGatherOp, AllReduceOp, AllToAllOp)

StableHLO tuple-collective ops support is limited to **single-operand** and **single-result**.
[MHLO ops](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td)
supports
**multi-operand** and **multi-result** which is in sync with XLA semantics and
horizontal scaling
[`all_reduce`](https://openxla.org/xla/operation_semantics#allreduce)
[`all_gather`](https://openxla.org/xla/operation_semantics#allgather) and
[`all_to_all`](https://openxla.org/xla/operation_semantics#alltoall) which
supports multi-operand and multi-result. `all_reduce` support is requested
in [#1370](https://github.com/openxla/stablehlo/issues/1370) is relied on by
PyTorch/XLA today via XlaBuilder ([ref](https://github.com/pytorch/xla/blob/1bbe333ad137ace6b8134db640c0b24c8c428db6/torch_xla/csrc/cross_replica_reduces.cpp#L156)).
`all_to_all` support is requested in
[#574](https://github.com/openxla/stablehlo/issues/574) and identified as a feature
gap.

## Proposed Specification Changes

View spec.md changes in this PR to view the diff vs original spec.

To view rich text of the spec, see:
[TanOp](https://github.com/abhigunj/stablehlo/blob/a3b4c1b69aff41e3175c1b4ccb6352bcadf1f79a/docs/spec.md#tan)
[CustomCallOp](https://github.com/abhigunj/stablehlo/blob/a3b4c1b69aff41e3175c1b4ccb6352bcadf1f79a/docs/spec.md#custom_call)
[AllGatherOp](https://github.com/abhigunj/stablehlo/blob/a3b4c1b69aff41e3175c1b4ccb6352bcadf1f79a/docs/spec.md#all_gather)
[AllReduceOp](https://github.com/abhigunj/stablehlo/blob/a3b4c1b69aff41e3175c1b4ccb6352bcadf1f79a/docs/spec.md#all_reduce)
[AllToAllOp](https://github.com/abhigunj/stablehlo/blob/a3b4c1b69aff41e3175c1b4ccb6352bcadf1f79a/docs/spec.md#all_to_all)
