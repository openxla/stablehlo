# StableHLO Compatibility

StableHLO is a backward compatible ML compute opset inspired by HLO/MHLO.
This document explains the kind and the extent of the compatibility guarantees
that StableHLO provides, based on the process established in
[the compatibility RFC](../rfcs/20220912-compatibility.md).

## Versions

The current version of StableHLO is **0.9.0**.

In the 0.x.x series, StableHLO is providing limited compatibility guarantees
as described in the Guarantees section. In H2 2023, we are planning to release
StableHLO 1.0.0, which will provide full compatibility guarantees - see the
Future work section for more details.

## Guarantees

**1 month of backward compatibility:** VHLO program serialized by an old version
of libStablehlo has the same semantics* when deserialized by a new version of
libStablehlo if these versions are built from openxla/stablehlo commits which
are less than 1 month apart.

**1 month of forward compatibility:** VHLO program serialized by a new version
of libStablehlo has the same semantics* when deserialized by an old version of
libStablehlo if these versions are built from openxla/stablehlo commits which
are less than 1 month apart, unless the program is using new features introduced
since the old version.

\* VHLO is a dialect that contains versioned snapshots of the StableHLO opset.
It can be obtained from the StableHLO dialect via
[--stablehlo-legalize-to-vhlo](../stablehlo/tests/stablehlo_legalize_to_vhlo.mlir),
and its semantics are defined by [the StableHLO spec](spec.md). See [vhlo.md](vhlo.md)
for more details and usage instructions.

## Out of scope

**Source compatibility** for C, C++ and Python APIs within libStablehlo is
an aspirational goal. At the moment, we don't offer source compatibility
guarantees, but please let us know if this is an important use case for you,
and we can have a discussion about supporting it
([#1247](https://github.com/openxla/stablehlo/issues/1247)).

**Compatibility for the StableHLO dialect** is not a goal. StableHLO dialect
is optimized for convenience, which means that it makes some choices which
are incompatible with stability (e.g. it only has the latest versions of
ops/attributes/types, it uses upstream attributes/types where possible, etc).
As a result, even though the StableHLO dialect can be serialized, we don't offer
any compatibility guarantees for the resulting payloads. When such guarantees
are needed, please use VHLO instead as described above.

## Tests

We have a compatibility suite in [stablehlo/testdata](../stablehlo/testdata)
that consists of several thousand files dumped from JAX tests. For every pull
request, we are testing that deserialization of the corresponding VHLO payloads
at HEAD produces programs which are syntactically identical to the corresponding
programs that were serialized by the latest release of libStablehlo.

## Future work

**5 years of compatibility guarantees.** In H2 2023, we are planning to release
StableHLO v1.0 which will implement high-priority improvements, including
cleaning up the frontend contract and providing a reference implementation.
Having obtained these improvements and resolved key specification compliance
issues, StableHLO will be ready for full compatibility guarantees - 5 years of
forward and backward compatibility. See [roadmap.md](roadmap.md) for details.

**Organize compatibility suite.** At the moment, the compatibility suite
is one directory with a ton of unstructured files. We are planning to triage and
organize it, making sure that it's organized, comprehensive and deduplicated
([#1240](https://github.com/openxla/stablehlo/issues/1240)).

**Use reference implementation.** At the moment, compatibility testing consists
of deserializing the compatibility suite serialized by older versions of
libStablehlo and making sure that deserialization produces syntactically
identical programs. We are planning to also use reference implementation in
these tests, relaxing the overly onerous requirement of syntactical identity
and comprehensively testing the reference implementation
([#1245](https://github.com/openxla/stablehlo/issues/1245)).
