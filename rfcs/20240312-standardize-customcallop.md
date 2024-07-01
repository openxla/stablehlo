# [RFC] Standardize CustomCallOp to extend backend_config to take a DictionaryAttr

Status: Approved<br/>
Initial version: 03/12/2024<br/>
Last updated: 03/15/2024<br/>
Discussion thread: [GitHub](https://github.com/openxla/stablehlo/pull/2097)

## Motivation

Several features have been added to MHLO in the past year, which frameworks want
to leverage and members of the community have made requests for them as well.
This includes: MHLO
[`custom_call`](https://github.com/tensorflow/mlir-hlo/blob/master/mhlo/IR/hlo_ops.td#L2483)
`backend_config` to take a `DictionaryAttr`.

The current StableHLO `custom_call` op does not support this feature. There are several
occurrences of users working around this gap in `custom_call` today, examples -
JAX uses [unregistered attributes](https://github.com/google/jax/blob/1ed27ecebb92e916b45601e3a107971170a4592b/jaxlib/hlo_helpers.py#L191)
to hold the `DictionaryAttr` or serializing the dictionary as a string
to pass around for the StableHLO `custom_call` op.

We propose standardizing the StableHLO `custom_call` op to extend `backend_config`
to take a `DictionaryAttr` so they can be used safely by the community without
workarounds. This will help to unify metadata under a single `DictionaryAttr` which
provides more stable serialization of `custom_call` metadata, a feature that is
desired by frameworks and compilers. Standardizing this feature to StableHLO
will benefit the entire ecosystem.
Note: `backend_config` will continue to accept strings as before.

Open tickets for this request: [#637](https://github.com/openxla/stablehlo/issues/637),
[#741](https://github.com/openxla/stablehlo/issues/741)

## Proposed Specification Changes

Please refer spec.md changes in this PR to view the diff vs original spec.
