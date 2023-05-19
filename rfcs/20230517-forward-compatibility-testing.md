# StableHLO Forward Compatibility Testing

## Background

StableHLO needs to verify both forward and backward compatibility. Currently we
check backward compatibility by versioning and serializing
`stablehlo_legalize_to_vhlo.mlir`, and need something similar for forward
compatibility. This document makes the following proposal:

**Proposal:** Add tests to compare bytes of bytecode file serialized at HEAD
with existing [backward compatibility bytecode test files](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/stablehlo_legalize_to_vhlo.0_9_0.mlir.bc).

## Requirements

Forward compatibility testing has the following requirements, to ensure that it
is both useful and practical:

* **R1.** Tests must detect changes in StableHLO that break forward compatibility.
* **R2.** Tests must detect LLVM revision bumps that break forward compatibility.
* **R3.** Tests must scale with releases and execute in a reasonable amount
of time.

## Proposed Design: Statically Test Forward Compatibility

In this design, forward incompatibilities are detected by serializing a file at
head and comparing it to a known-good file serialized at the previous release.
This approach operates on the assumption that at HEAD we are able to produce
a byte-identical serialized artifact, excluding custom header info and debug
locations.

Detecting forward incompatiblities can be accomplished by adding the following
`RUN` line to the versioned [`stablehlo_legalize_to_vhlo.0_X_0.mlir`](https://github.com/search?q=repo%3Aopenxla%2Fstablehlo+path%3A**%2Fstablehlo_legalize_to_vhlo.0_*&type=code)
test files.

```bash
diff %s.bc <(stablehlo-opt %s | stablehlo-translate --serialize --target=0.X.0)
```

Passing through `stablehlo-opt` in this example removes debug info. An option
to `stablehlo-translate` can be added to strip debug info and header info. This
test should be run in CI on all PRs to prevent merging incompatible changes in
StableHLO (_R1_) and LLVM revision bumps (_R2_).

The StableHLO process for [Contributing Incompatible Changes](https://github.com/openxla/stablehlo/blob/main/docs/vhlo.md#add-versioned-serialization-test)
requires a versioned test be serialized for each compatibility breaking change,
which means that we can test forward compatibility between HEAD and each minor
release of StableHLO. Given that all changes to the StableHLO opset, as well as
changes to serialization machinery like targeting a newer version of the MLIR
bytecode format, require bumping the minor version, there is a low chance of
causing forward incompatibilities between different patch versions of StableHLO
(i.e. `0.10.1 → 0.10.2`). Ensuring compatibility between HEAD and all minor
releases (`0.9.0`, `0.10.0`, etc.) should provide thorough enough coverage of the
two mentioned sources of forward incompatibilities.

Other benefits of this approach include that it is lightweight (_R3_), can be run
locally to detect compatibility issues during development, and it fits nicely into
existing CI build-and-test jobs. If the underlying byte-identical assumption is
invalidated, we can fall back to the alternate design in a hybrid testing approach,
since byte-comparison will safely identify all forward incompatibilities, with a
slight risk of false positives that should be tested using the alternate design.

## Alternate Design: Serialize at HEAD, Deserialize at Target Release

This design is explored more in a [previous commit of this RFC](https://github.com/openxla/stablehlo/blob/0792eb75e85c54f9d106878569b088d03c568b70/rfcs/20230517-forward-compatibility-testing.md#preferred-design-serialize-at-head-deserialize-at-target-release),
and is summarized in this version for brevity. If byte-equality cannot be safely
relied on, then we may need to use this approach, or a hybrid of the two. This
design can be summarized as:

* Maintain bytecode files serialized at HEAD, which are re-serialized before
  each StableHLO release (~2x/wk): `stablehlo_legalize_to_vhlo.0_X_0.mlirbc.HEAD`
* Add CI jobs that build StableHLO/MLIR at every tagged minor release
  (`v0.9.0`, `v0.10.0`, etc.) and check out the `0_X_0.mlirbc.HEAD` file that
  corresponds to the tagged release to deserialize and run FileCheck tests on.

This approach is more resilient to compatible changes in the bytecode format.
The downsides of this approach include that it cannot easily be tested locally,
requiring CI jobs. As such it doesn't work out of the box in other environments,
like projects that rely on StableHLO and want to run StableHLO tests. And
lastly it is more costly and time consuming, as it requires building several
releases of StableHLO and LLVM/MLIR.

Given the pros/cons of these two designs, this CI should only be used as needed.
If there are changes that cause bytecode differences that cannot be tested
statically, this infrastructure can be used. Otherwise, when possible, the
static verification should be used.
