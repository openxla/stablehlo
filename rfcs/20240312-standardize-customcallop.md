# [RFC] Standardize CustomCallOp to extend backend_config to take a DictionaryAttr

Status: Review<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/12/2024<br/>
Discussion thread: [GitHub](add PR Link)

## Motivation

Several features have been added to MHLO in the past year, which frameworks want
to leverage and members of the community have made requests for them as well.
This includes:  `CustomCallOp` to support backend_config to take a DictionaryAttr.
This feature is being used today with various workarounds -- unregistered
attributes, serializing dictionary as string, custom_calls. None of these
approaches are stable, so we propose adding these ops/features to the StableHLO
spec so they can be used safely by the community.

### CustomCallOp with typed FFI

StableHLO `custom_call` op supporting `backend_config` dictionary will help to
unify metadata under single `mlir::DictionaryAttr`. This provides more stable
serialization of `custom_call` metadata, a feature that is desired by frameworks
and compilers. There are several occurrences of users working around this gap in
`custom_call` today, either as an unregistered attribute
([example](https://github.com/google/jax/blob/1ed27ecebb92e916b45601e3a107971170a4592b/jaxlib/hlo_helpers.py#L191)),
or a serialized dictionary string. Standardizing this feature to StableHLO will
benefit the entire ecosystem. We propose to support DictionaryAttr for
`backend_config`, the same as what [MHLO custom_call op](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L2483)
already supporting. Open tickets for this request: [#637](https://github.com/openxla/stablehlo/issues/637),
[#741](https://github.com/openxla/stablehlo/issues/741)

## Proposed Specification Changes

View spec.md changes in this PR to view the diff vs original spec.

To view rich text of the spec, see:
[CustomCallOp](https://github.com/openxla/stablehlo/blob/f8d6756c70dc5301d5be88d1ca378d1429943e0c/docs/spec.md#custom_call)
