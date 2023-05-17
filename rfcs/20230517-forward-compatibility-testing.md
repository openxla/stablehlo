# StableHLO Forward Compatibility Testing

## Background

StableHLO needs to verify both forward and backward compatibility via unit
tests. Currently we check backward compatibility by versioning
`stablehlo_legalize_to_vhlo.mlir`. This document makes the following proposals:

* **Proposal 1:** Maintain bytecode at HEAD to use for forward compatibility
testing.
* **Proposal 2:** Add CI which builds at tagged release commits to check
forward compatibility.
* **Proposal 3:** Run CI on all supported minor releases of StableHLO
(`v0.9.0`, `v0.10.0`, etc.).

## Requirements

Forward compatibility testing has the following requirements, to ensure that it
is both useful and practical:

* **R1.** Tests must detect changes in StableHLO that break forward compatibility.
* **R2.** Tests must detect LLVM revision bumps that break forward compatibility.
* **R3.** Test job must scale with releases and execute in a reasonable amount
of time.

## Preferred Design: Serialize at HEAD, Deserialize at Target Release

_Note: Section only discusses how to test a generic release `0.X.0`. Discussion
on what values of `X` to test for are discussed in the later section "What
releases should be tested?"_

### Serialize and Commit Test Files at HEAD

Prior to tagging each StableHLO release, which occurs ~2x per week, generate
versioned forward compatibility tests using the
`stablehlo_legalize_to_vhlo.0_X_0.mlir` test file as follows:

```bash
stablehlo-translate --serialize --target=0.X.0
  stablehlo_legalize_to_vhlo.0_X_0.mlir >
    stablehlo_legalize_to_vhlo.0_X_0.mlir.bc.HEAD
```

The output will be a bytecode file that targets `0.X.0` and should be runnable
at release `0.X.0`. Once these files are generated and checked in, we can add
the following RUN line to the source `.mlir` file to test serialization,
deserialization, and FileCheck comparisons:

```bash
// RUN: stablehlo-translate --deserialize %s.bc.HEAD |
  stablehlo-translate --serialize --target=0.X.0 |
    stablehlo-opt --mlir-print-op-generic | FileCheck %s
```

**Proposal 1:** Maintain bytecode at HEAD to use for forward compatibility
testing.

### Add Forward Compatibility CI

A CI job which blocks pull request merges to main can be added to ensure
forward compatibility of PRs. To test version `0.X.0` the CI job does the
following:

* Checkout `openxla/stablehlo` at tag `v0.X.0`.
* Checkout file at HEAD `stablehlo_legalize_to_vhlo.0_X_0.mlir.bc.HEAD`
* Append FileCheck line to `stablehlo_legalize_to_vhlo.0_X_0.mlir`
* Perform the existing build-and-test CI workflow.

This will rebuild LLVM at the proper revision, rebuild StableHLO at the tagged
release, and bring in the forward compatibility test to the CI workspace. Any
failures in this test file will indicate a forward incompatibility at HEAD. A
prototype of this CI job can be seen [here](https://github.com/GleasonK/stablehlo/pull/33).

**Proposal 2:** Add CI which builds at tagged release commits to check forward
compatibility.

## What releases should be tested?

Testing every single release would be sound, but more expensive and time
consuming than needed, and doesn't scale well with multiple releases a week
(_R3_). Given that all changes to the StableHLO opset (_R1_), as well as
changes to serialization machinery like targeting a newer version of the MLIR
bytecode format (_R2_), require bumping the minor version, there is a low
chance of causing forward incompatibilities between different patch versions of
StableHLO (i.e. `0.10.1 → 0.10.2`). Ensuring compatibility between HEAD and all
minor releases (`0.9.0`, `0.10.0`, etc.) should provide thorough enough
coverage of the two mentioned sources of forward incompatibilities.

Currently, this means that we only serialize a file at HEAD for all supported
minor releases, and CI will check out each tagged minor release for testing.

**Proposal 3:** Run CI on all supported minor releases of StableHLO
(`v0.9.0`, `v0.10.0`, etc.).

## Alternate Design: Statically Test Forward Compatibility

Statically test forward compatibility by comparing the bytes produced by a
writer, excluding the header and comparing the binary segment byte-for-byte.
This is a good idea, but doesn’t work in practice. The bytecode may change but
still be readable by a previous version, which is the case currently. Files
serialized at `StableHLO@HEAD` are different from `StableHLO@v0.9.0`, but both
produce identical IR when read at `v0.9.0`.
