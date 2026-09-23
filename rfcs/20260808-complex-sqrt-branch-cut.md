# [RFC] Specify the complex `sqrt` branch cut at signed zero

Status: Review<br/>
Initial version: 08/08/2026<br/>
Last updated: 08/08/2026<br/>
Discussion thread: [Issue #2988](https://github.com/openxla/stablehlo/issues/2988)

## Summary

This RFC proposes defining the value of complex `stablehlo.sqrt` on the
negative real axis when the imaginary component of the operand is positive or
negative zero.

For finite `x < 0`, the proposed requirement is:

```text
sqrt(complex(x, +0.0)) = complex(+0.0, +sqrt(-x))
sqrt(complex(x, -0.0)) = complex(+0.0, -sqrt(-x))
```

The requirement applies when a signed zero is present at the `sqrt` operand. It
does not require unrelated or preceding StableHLO operations to preserve signed
zero.

## Motivation

The current specification describes complex `sqrt` only as "complex square
root". This leaves the two sides of its negative-real branch cut
underspecified.

The sign of a zero imaginary component identifies the side from which the
branch cut is approached. Respecting that sign follows the principal square
root convention that is continuous from the corresponding half-plane. It also
preserves the expected conjugation relation at the cut:

```text
sqrt(conj(z)) = conj(sqrt(z))
```

This distinction is observable in array APIs. For example, NumPy returns
`+2j` for `sqrt(-4+0j)` and `-2j` for `sqrt(-4-0j)`. The current JAX/XLA path
returns `+2j` for both inputs, as reported in
[jax-ml/jax#39110](https://github.com/jax-ml/jax/issues/39110).

There is also an inconsistency within StableHLO today. The reference interpreter
uses the host complex square root and preserves the sign for `-4-0j`, while the
generated `stablehlo-complex-math-expander` lowering selects the positive
imaginary result for both signed-zero inputs. The generated fixtures use
approximate comparison and an MPMath-backed reference path that does not
independently assert the sign of zero, so they pass despite the mismatch. The
lowering was introduced in
[StableHLO#2679](https://github.com/openxla/stablehlo/pull/2679) and currently
selects the sign with an ordered comparison that treats `-0.0` like `+0.0`.

## Proposed specification

Change the complex-number case in the `sqrt` semantics to specify the branch
convention for finite operands on the negative real axis. When the real
component is finite and negative and the imaginary component is zero, the
result has a positive-zero real component and an imaginary component whose sign
matches the input imaginary component.

No other part of the `sqrt` interface changes. Operand and result types,
quantized behavior, and result-accuracy behavior remain unchanged. This RFC does
not further specify behavior for infinities or NaNs.

## Conformance and implementation

After approval, implementation work should include:

1. Update `docs/spec.md` with the signed-zero branch-cut requirement.
2. Add exact sign-bit conformance cases for complex64 and complex128 to the
   reference and complex-math-expander tests. Approximate equality is not
   sufficient because `+0.0` and `-0.0` compare equal.
3. Update the generated complex `sqrt` lowering so that `-0.0` selects the
   negative side of the branch cut.
4. Verify that the corrected StableHLO expander is integrated into XLA, then
   add or enable a JAX regression test after `jaxlib` incorporates that XLA
   revision.

StableHLO currently treats generated math implementations as immutable and
sources them from
[`functional_algorithms`](https://github.com/pearu/functional_algorithms).
[functional_algorithms#118](https://github.com/pearu/functional_algorithms/pull/118)
contains a candidate generator change and focused tests, but that repository has
not published a release containing the change. Before implementation, the
StableHLO maintainers should confirm whether the normal generator release path
is required or whether another source path is acceptable if the dependency is
not actively maintained.

## Compatibility

This proposal narrows behavior that is currently underspecified. Programs that
observe the sign of the result on the lower side of the negative-real branch
cut will change on implementations that currently collapse both inputs to the
upper-side result.

No wire-format change is proposed. Operation syntax, types, attributes, and
verifier behavior remain unchanged. Maintainers should confirm whether this
semantic clarification has VHLO versioning implications. Its compatibility
effect is limited to the numerical semantics of existing complex `sqrt`
operations.

## Alternatives considered

### Keep the behavior in JAX

A `jax.numpy.sqrt` wrapper can restore NumPy compatibility for JAX users. It
would not define the behavior for other StableHLO producers or backends, and it
would leave the StableHLO reference interpreter and complex-math expander in
disagreement.

### Change the lowering without changing the specification

An implementation-only fix would remove the observed inconsistency, but the
same ambiguity could recur in another backend because the required branch
convention would remain unstated.

### Require general signed-zero preservation

A general guarantee for signed zero across preceding arithmetic would be much
broader than the reported problem and may constrain optimizations unrelated to
complex `sqrt`. This RFC only requires respecting the sign that reaches the
operation boundary.

### Specify other branch-cut functions at the same time

Other complex functions may have similar branch-cut questions. They should be
evaluated separately. This proposal is intentionally limited to the reproduced
`sqrt` inconsistency.
