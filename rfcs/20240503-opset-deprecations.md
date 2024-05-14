# RFC: StableHLO v1.0 Opset Deprecations & Cleanups

Author: gleasonk<br/>
Last Modified: 5/13/24<br/>
Status: Approved<br/>

## Background

This doc covers a list of opset cleanups that we want to do for StableHLO v1.0.
Most of these ops were never spec’ed and therefore have no formal compatibility
guarantees, per the [Unspecced Features compatibility exemption][compat-out-of-scope],
however we can provide some backward / forward compatibility for most of them.

This doc will propose futures for the ops intentionally omitted from the spec,
including:

- [“Not in HLO” ops][not-in-HLO] ([#3](https://github.com/openxla/stablehlo/issues/3)):
`broadcast`, `create_token`, `cross-replica-sum`, `dot`, `einsum`,
`torch_index_select`, `unary_einsum`, `trace` ([#604](https://github.com/openxla/stablehlo/issues/604)).
- [Dynamism RFC P4](https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md#p4)
opset updates: `real_dynamic_slice` vs `dynamic_slice`.
- Potentially unused ops like stateful `rng` [#597](https://github.com/openxla/stablehlo/issues/597)
and `map`.
- Tuple Ops and type, including `get_tuple_element` and `tuple` op, along with
`tuple` type support in `custom_call` ([#598](https://github.com/openxla/stablehlo/issues/598)).
- Features explicitly denoted as "to be removed" in the spec, such as `tuple`
type ([#598](https://github.com/openxla/stablehlo/issues/598)), comparisons of
complex types ([#560](https://github.com/openxla/stablehlo/issues/560)), and
`convolution` op's `window_reversal`, which are unreliable or unused.

In general (unless the op is unused and can be trivially deleted), the
deprecation steps will be as follows:

1. Migrate framework uses of redundant ops.
1. Block serialization of deprecated ops once frameworks migrated.
1. Migrate uses of the ops to the supported StableHLO op (add builder methods).
1. Change VHLO legalization to upgrade to the supported op for compatibility.
1. Remove the redundant StableHLO op.
1. Remove redundant op from VHLO after 6 months.

## Proposed Opset Changes

### P0: Delete `CreateTokenOp` and `TraceOp`

These ops are both unused as far as we can tell. They can be trivially deleted.

### P1: Deprecate `BroadcastOp`, `DotOp`, `UnaryEinsumOp`

These ops are all a trivial subset of features of another op. I.e. BroadcastOp
can be represented using BroadcastInDim, DotOp with DotGeneralOp, UnaryEinsum
with [`einsum` lowering][einsum-lowering].
These ops will follow the formal deprecation process listed above.

Helper methods can be added to the support op for compatibility, something like:
`isLhsBroadcast`, `isSimpleDot`, `isUnaryEinsum`.

### P2: Deprecate `RealDynamicSliceOp`, Enhance `DynamicSliceOp`

In terms of naming `stablehlo.dynamic_slice` is more in-model than
`real_dynamic_slice`. However in terms of functionality, per
[Dynamism RFC P4](https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md#p4)
the behavior of `real_dynamic_slice` is correct. We propose to enhance
`dynamic_slice_op` to have an identical feature set as `real_dynamic_slice`, and
deprecate `real_dynamic_slice`. This change will be done with full
forward and backward compatibility.

One could make the argument that `dot` is a more proper name than `dot_general`,
and I'm happy to go down that route, but it will likely cause a good deal of
code churn in community repos. Interested in feedback here.

### P3: Move `CrossReplicaSumOp` to CHLO

The `cross-replica-sum` op  (hyphens not a typo), is just sugar for an
`all-reduce` op. Even in the XlaBuilder's [xla::CrossReplicaSum][CRS]
implementation this op is decomposed into an all reduce. We could just remove
this op, and eventually we may, but we propose to move it to CHLO in the short
term since frameworks map to this op, and this will keep the refactoring fairly
trivial.

### P4: Deprecate `MapOp`, `RngOp`, `EinsumOp` `TorchIndexSelectOp`, Tuple support

**Feedback Requested:** These opset changes are pending community feedback.

These are all ops that seem to have very limited use in StableHLO. It would be
great to remove them all or move them to CHLO, as opposed to providing long term
compatibility on ops that aren't needed.

In the interim, we only plan to guarantee the existing 6 month compatibility
guarantees until these ops' futures are more clearly known.

- **MapOp** is unused as far as we can tell, including in HLO. Its uses tend to
be just for a region to mimic a composite, which is no longer needed after the
addition of the `stablehlo.composite` op. This op likely can be removed.
- **RngOp** is stateful, and there is a better alternative in
`RngBitGeneratorOp`. More work needs to be done to determine if all uses of this
op can be safely migrated to the alternative.
- **EinsumOp** can likely be moved to CHLO, the [xla::Einsum][einsum] method is
similarly a decomposition. It is unclear how necessary this abstraction is for
linalg lowerings though.
- **TorchIndexSelectOp** can also likely be moved to CHLO. There is an existing
[lowering to `gather`][torch-index-select] which can be used for a
decomposition. However, similar to `einsum`, it is unclear how necessary this
abstraction is to the community.
- **Tuple Support** includes `get_tuple_element` and `tuple` ops, along with,
support for `tuple` type in `custom_call` ([#598](https://github.com/openxla/stablehlo/issues/598)).
The use of tuples in MLIR is limited, and these are mostly kept around for
interop with XLA and other dialects.

Interested in feedback on any of the above proposals, or ideas for how to keep
these changes from being too invasive to community projects!

[compat-out-of-scope]: https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md#out-of-scope
[not-in-HLO]: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#:~:text=%22Not%20in%20HLO%22,-category%20of%20StableHLO
[CRS]: https://github.com/openxla/xla/blob/6cc24d8548094b3fc94dacc569fc6959227ae28b/xla/client/xla_builder.cc#L3619
[einsum]: https://github.com/openxla/xla/blob/8371ea90202d9ca1cb1148237a1a1ef3620b354a/xla/client/lib/matrix.cc#L386
[einsum-lowering]: https://github.com/openxla/xla/blob/6cc24d8548094b3fc94dacc569fc6959227ae28b/xla/mlir_hlo/mhlo/IR/mhlo_canonicalize.td#L30
[torch-index-select]: https://github.com/openxla/xla/blob/8371ea90202d9ca1cb1148237a1a1ef3620b354a/xla/mlir_hlo/mhlo/transforms/legalize_torch_index_select_to_gather/legalize_torch_index_select_to_gather.cc#L45
