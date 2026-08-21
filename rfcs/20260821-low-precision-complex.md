# [RFC] Support f16- and bf16-based complex types in StableHLO

Status: In Review<br/>
Initial version: 08/21/2026<br/>
Last updated: 08/21/2026<br/>
Discussion thread: TBD

## Overview

This RFC proposes extending the set of complex types supported by StableHLO
from:

```text
complex<f32>
complex<f64>
```

to:

```text
complex<f16>
complex<bf16>
complex<f32>
complex<f64>
```

The proposal uses the existing MLIR builtin `ComplexType`; it does not add a
new StableHLO type. It extends the set of floating-point component types that
StableHLO accepts inside `complex<T>`.

This is a StableHLO type-system and serialization proposal. It does not require
XLA, JAX, or any particular backend to execute the new types, and it does not
promise native kernels, a memory ABI, or improved performance.

## Motivation

[openxla/stablehlo#1794](https://github.com/openxla/stablehlo/issues/1794)
requests support for 16-bit FFT input and output types. Today, the StableHLO
verifier prevents such programs even though MLIR can represent both
`complex<f16>` and `complex<bf16>`.

There are two independent concerns:

1. Whether StableHLO can represent and specify a low-precision complex value.
2. Whether a particular consumer or backend can execute that value efficiently.

StableHLO should decide the first concern based on the coherence of its type
system and semantics rather than on the current implementation status of one
consumer. Consumers that do not support the new types may continue to reject
them with a diagnostic until they add a lowering, legalization, or native
implementation.

The body of #1794 suggests allowing any floating-point component type. This
RFC intentionally narrows that suggestion to f16 and bf16, the two widely used
16-bit floating-point formats, in addition to the existing f32 and f64 types.
Supporting only `bf16` would not address the issue's original `f16` request,
while allowing every member of StableHLO's broader floating-point type set
would also introduce complex types with FP4, FP6, FP8, and other specialized
component formats. Those formats have distinct representability and
compatibility questions and should not be included implicitly in this
proposal.

## Current behavior

The current StableHLO specification lists only `complex<f32>` and
`complex<f64>` as supported complex types.

The implementation reflects this restriction in multiple places:

- `HLO_Complex` is defined as `Complex<HLO_Float32Or64>`.
- The operands of `stablehlo.complex` are restricted to f32/f64 tensors.
- RFFT type inference has a hand-written check that requires f32 or f64 input.
- The StableHLO test suite contains a negative test requiring f16 RFFT input to
  be rejected.

Changing only the RFFT check would be incomplete: RFFT could infer a type that
the common StableHLO complex constraint and `stablehlo.complex` still reject.

VHLO currently represents MLIR builtin complex types with `ComplexV1`, whose
verifier only checks that the component is a VHLO type. This is a shallow
structural check: the representation can therefore spell component
combinations that were never in the historical StableHLO complex type domain.
Historically valid `ComplexV1` values are limited to f32 and f64 components.

The current VHLO version-conversion pass has no type-to-type version
conversions because no VHLO type has previously required one. It recursively
checks whether nested types are legal for the target version, but the type
converter itself is not target-version-aware. The versioned type introduced by
this RFC therefore requires new conversion infrastructure rather than only an
additional type definition.

## Proposal

### Supported complex component types

StableHLO will define an explicit complex component type set:

```text
ComplexComponent ::= f16 | bf16 | f32 | f64
ComplexType      ::= complex<ComplexComponent>
```

The implementation should use a dedicated ODS constraint for this set rather
than define `HLO_Complex` in terms of the full `HLO_Float` constraint. This
keeps future additions to `HLO_Float` from automatically changing the
StableHLO complex type system without review.

No mixed-component complex type is introduced. A complex value continues to
have real and imaginary components with the same element type.

### Effect on existing operations

This proposal does not add an operation and does not add complex semantics to
an operation that currently excludes complex types.

For every existing operation whose specified input or result type domain
already includes a complex type, `complex<f16>` and `complex<bf16>` become
additional valid instantiations of that existing type domain. This RFC
proposes no operation-specific exception. If review identifies an operation
whose semantics cannot support either new instantiation, that exception must
be specified in a revision of this RFC rather than chosen during
implementation.

This consequence is intentional. Restricting the new types to FFT alone would
allow a low-precision complex value to be produced but prevent it from flowing
through the same basic construction, extraction, arithmetic, conversion, and
structural operations as other StableHLO complex values.

The observable result type of an operation remains the type written in the
StableHLO program. This RFC does not insert implicit promotion to f32 and does
not change `complex<f16>` or `complex<bf16>` into another IR type. A consumer
may use wider intermediate precision where permitted by the existing operation
semantics and accuracy rules, but the declared input and result types remain
unchanged.

The normative affected surface is grouped as follows:

- Direct construction and extraction operations, including `complex`, `real`,
  and `imag`, accept the two new instantiations.
- Existing parametric complex arithmetic and transcendental operations accept
  them under their existing semantics and accuracy contracts. This includes
  the direct complex-domain uses in `abs`, `atan2`, `cbrt`, `cosine`, `divide`,
  `exponential`, `exponential_minus_one`, `log`, `log_plus_one`, `logistic`,
  `negate`, `power`, `remainder`, `rsqrt`, `sign`, `sine`, `sqrt`, `subtract`,
  `tan`, and `tanh`, as well as operations such as `add` and `multiply` whose
  generic tensor domains already include complex element types.
- Existing complex linear-algebra operations, including `cholesky` and
  `triangular_solve`, accept the two new instantiations.
- FFT, IFFT, RFFT, and IRFFT accept the relationships specified below.
- Constants, `iota`, conversion, data-movement, shape-preserving, control-flow,
  and structural operations continue to accept complex values wherever their
  current type domains already do so.
- A supported complex type may occur wherever an existing StableHLO container
  admits a complex element or nested value, including tensors, buffers,
  tuples, futures, function signatures, region arguments, and custom calls.

Because `HLO_Complex` is used transitively, the implementation audit must cover
the complete ODS constraint graph rather than only literal `HLO_Complex`
occurrences. In particular, derived constraints such as `HLO_Tensor`,
`HLO_NonQuantizedTensor`, `HLO_Buffer`, `HLO_Tuple`, `HLO_CustomCallValue`, and
their static-shape or unranked variants expand with the common complex
component set. Every custom verifier and inference path reached through these
constraints must be checked against the specification above.

This expansion does not create separate kinds of complex type. It also does
not define the physical layout or ABI of a buffer containing a low-precision
complex element; StableHLO buffer acceptance remains an abstract type-system
property under the existing buffer contract.

### `complex`, `real`, and `imag`

For every supported complex component type `T`:

```text
stablehlo.complex : (tensor<...xT>, tensor<...xT>)
                 -> tensor<...xcomplex<T>>

stablehlo.real    : tensor<...xcomplex<T>> -> tensor<...xT>
stablehlo.imag    : tensor<...xcomplex<T>> -> tensor<...xT>
```

The existing shape and same-element-type constraints continue to apply.

In particular, the following remain invalid:

- constructing a complex value from one f16 tensor and one bf16 tensor;
- declaring a `complex<f32>` result for f16 components;
- declaring a real or imaginary result whose component type differs from the
  input complex component type.

### FFT type relationships

For `T` in `{f16, bf16, f32, f64}`, the FFT type relationships are:

```text
FFT:   complex<T> -> complex<T>
IFFT:  complex<T> -> complex<T>
RFFT:          T  -> complex<T>
IRFFT: complex<T> -> T
```

The existing FFT rank, shape, and `fft_length` constraints are unchanged.

The hand-written RFFT type check should use the same supported complex
component predicate as the common complex type constraint, avoiding a second
component-type list that can drift from the specification.

### Constants and conversion

Complex constants with f16 or bf16 components follow the existing StableHLO
complex constant definition: the real and imaginary literals are interpreted
using the floating-point semantics of the component type.

`stablehlo.convert` continues to use its existing specified conversion
semantics. This RFC makes the new complex types valid source or result types in
cases where the existing `convert` type domain includes complex types; it does
not introduce an implicit conversion or a new fallback rule.

## Compatibility and VHLO

Low-precision complex support is a new StableHLO feature and therefore requires
a StableHLO minor version bump. The implementation must use the next available
minor version at the time it lands. If implemented immediately after StableHLO
1.20.0, the feature version would be 1.21.0.

The required observable compatibility behavior is:

- serialization targeting the feature version or a later version succeeds;
- serialization of a program using `complex<f16>` or `complex<bf16>` to an
  earlier target version fails with a diagnostic;
- no compatibility path may silently promote the value to `complex<f32>` or
  decompose it into unrelated tensor types;
- programs using only `complex<f32>` and `complex<f64>` retain their existing
  ability to target older supported StableHLO versions;
- current consumers remain able to deserialize portable artifacts containing
  historical f32/f64 complex programs.

### Proposed VHLO representation

VHLO currently has `ComplexV1`, available since 0.9.0. Its representation is
structurally capable of containing any VHLO component type, but historical
StableHLO versions only specified `complex<f32>` and `complex<f64>`. Reusing
`ComplexV1` for f16/bf16 without an additional version boundary would make a
new StableHLO feature appear incorrectly available to old target versions.

This RFC requires adding `ComplexV2` at the new feature version. The versioned
contract is normative:

- `ComplexV1` remains the historical VHLO representation. Its version range is
  closed at the minor version immediately preceding the feature version, as in
  the standard VHLO version-split pattern. The valid portable StableHLO domain
  represented by historical V1 artifacts is f32- and f64-based complex types.
- The existing shallow parser and verifier behavior of `ComplexV1` is not
  tightened. A V1 value with another VHLO component type may remain
  structurally representable, but it was not a valid historical portable
  StableHLO artifact and must not become one after this feature lands.
- `ComplexV2`, available from the feature version through the current version,
  represents exactly f16-, bf16-, f32-, and f64-based complex types. Its
  verifier enforces this closed component set.
- FP8, integer, and all other component types are invalid inhabitants of V2 and
  are not valid portable StableHLO inhabitants of V1.

StableHLO-to-current-VHLO conversion produces `ComplexV2` for every supported
StableHLO complex type, including `complex<f32>` and `complex<f64>`. Conversely,
conversion of historical VHLO to the current version upgrades every valid
`ComplexV1<f32>` or `ComplexV1<f64>` to the corresponding `ComplexV2` before
converting to StableHLO. This keeps the current representation uniform rather
than partitioning complex types by component width.

When targeting a version before the feature version, each `ComplexV2<f32>` or
`ComplexV2<f64>` is downgraded to the corresponding `ComplexV1`. A
`ComplexV2<f16>` or `ComplexV2<bf16>` has no semantics-preserving downgrade and
must cause conversion to fail. When upgrading V1 to the current version, any
component other than f32 or f64 must likewise cause conversion to fail. In
particular, shallow structural representability does not make
`ComplexV1<f16>` or `ComplexV1<bf16>` a historical StableHLO type.

### Recursive type conversion requirements

The V1-to-V2 upgrade and V2-to-V1 downgrade must be target-version-aware and
recursive. Applying the conversion only to an operation's immediate result
types would leave invalid versioned types hidden in containers or signatures.
The conversion and legality checks must cover every type-bearing location
supported by VHLO, including:

- `RankedTensorV1`, `UnrankedTensorV1`, and `RankedBufferV1` element types;
- `TupleV1` elements, nested tuples, and `FutureV1` element types;
- `FunctionV1` inputs and results;
- operation operands and results, function signatures, and region block
  arguments;
- `TypeV1Attr`, `TensorV1Attr`, `FloatV1Attr`, `IntegerV1Attr`, and every other
  attribute that carries a type;
- ranked-tensor encodings and nested array or dictionary attributes; and
- the storage and expressed types of `UniformQuantizedV1` and
  `UniformQuantizedPerAxisV1`, which must be traversed even though a complex
  inhabitant may be rejected by their independent semantic constraints.

If any nested occurrence cannot be converted for the requested target, the
entire version conversion must fail before bytecode is written. The diagnostic
must identify the target version and the unsupported complex type. No path may
partially convert the module, silently promote the component, or leave a
new-version type embedded in an old-version artifact.

### Conversion algorithm and invariants

The version-conversion implementation must follow these observable steps:

1. Parse and validate the target version before constructing or applying the
   type converter. Type conversion decisions must therefore have access to the
   target version.
2. For a target at or after the feature version, convert
   `ComplexV1<f32/f64>` to `ComplexV2<f32/f64>` and reject every other V1
   component during upgrade. Leave an already valid V2 unchanged.
3. For a target before the feature version, convert `ComplexV2<f32/f64>` to V1,
   reject `ComplexV2<f16/bf16>`, leave `ComplexV1<f32/f64>` unchanged, and
   reject every other V1 component. A structurally representable but
   nonhistorical V1 value must never be emitted into an old-version artifact.
4. Rebuild every enclosing VHLO type and type-bearing attribute recursively
   when a nested component changes. This includes rebuilding tensor constants
   with their converted `TensorV1Attr` type while preserving their value.
5. Update operation operands and results, function signatures, region block
   arguments, and type-bearing attributes consistently. An operation whose
   version is otherwise legal still requires a generic identity rewrite, or an
   equivalent pre-conversion step, when one of its types changes.
6. Apply existing operation-version rewrites and then run a final recursive
   legality check over all operations, regions, types, and attributes.

The transformation is all-or-nothing from the serializer's perspective. Any
failed type or attribute conversion fails the pass, and serialization must not
emit bytecode from the partially converted IR.

## Consumer behavior

StableHLO verifier acceptance is not a claim that every consumer can execute
the new types.

A consumer that lacks support may:

- reject a module containing the new types with a clear unsupported-type
  diagnostic;
- apply a separately specified legalization;
- implement the types natively.

This RFC does not prescribe which option a consumer must choose. In particular,
it does not make XLA support a prerequisite for StableHLO type validity.

## Verification and testing

The implementation must include positive tests for:

- parsing and printing `complex<f16>` and `complex<bf16>` in StableHLO
  programs;
- `stablehlo.complex`, `stablehlo.real`, and `stablehlo.imag` type relations;
- an acceptance case for every existing operation with a direct complex-domain
  numeric, generator, or linear-algebra constraint, including `iota`, plus
  representative cases for every transitive structural and container
  constraint family listed above;
- FFT, IFFT, RFFT, and IRFFT for both f16 and bf16 component types;
- exact return-type inference, including dynamic-shape cases;
- StableHLO-to-current-VHLO-to-StableHLO round trips for both `complex<f16>`
  and `complex<bf16>`;
- serialization and deserialization at the new version for both
  `complex<f16>` and `complex<bf16>`;
- deserialization of historical `ComplexV1<f32>` and `ComplexV1<f64>` artifacts
  through their upgrade to current `ComplexV2`;
- serialization of current `ComplexV2<f32>` and `ComplexV2<f64>` through their
  downgrade to `ComplexV1` for an older target version;
- recursive upgrade and downgrade through ranked and unranked tensors, ranked
  buffers, tuples, futures, function signatures, operation types, region block
  arguments, tensor encodings, quantized-type parameters, `TypeV1Attr`,
  `TensorV1Attr`, `FloatV1Attr`, `IntegerV1Attr`, and nested array or dictionary
  attributes; and
- continued old-version serialization of f32/f64 complex programs.

The implementation must include negative tests for:

- complex values with integer component types;
- at least one floating-point component type outside this RFC, such as an FP8
  type;
- rejection of `ComplexV1<f16>`, `ComplexV1<bf16>`, and every other
  nonhistorical V1 component during current-version upgrade, deserialization,
  and conversion to an old target, both directly and when nested in every
  supported type-bearing container, attribute, and quantized-type location
  listed above, without tightening the historical shallow V1 parser or
  verifier;
- `ComplexV2` with an FP8, integer, or other unsupported component;
- mismatched f16/bf16 inputs to `stablehlo.complex`;
- mismatched real/complex component types in FFT input and result types;
- attempting to serialize a program containing `complex<f16>` or
  `complex<bf16>` to the version immediately before the feature version;
- failed downgrade of a low-precision complex type nested in each supported
  container category; and
- failure before bytecode emission, with a diagnostic that names both the
  requested target version and the unsupported nested type.

The VHLO compatibility suite must include the new versioned textual fixture and
its bytecode fixture, following the existing VHLO checklist.

Reference-interpreter numerical tests may be delivered in a separate
StableHLO-only follow-up if the required tensor storage and constant
materialization changes would obscure review of the type-system and
compatibility change. If maintainers require reference execution as an opset
acceptance criterion, that follow-up can be stacked or folded into the
implementation change without adding XLA or backend work.

## Non-goals

This RFC does not propose:

- an XLA primitive type or StableHLO-to-XLA bridge change;
- JAX dtype, promotion, tracing, or lowering changes;
- portable decomposition into real and imaginary tensor planes;
- CPU, GPU, TPU, or accelerator kernels;
- FFT library integration;
- a memory layout, storage ABI, or claim that a value occupies four bytes in a
  particular runtime;
- mandated accumulation precision or a new numerical-accuracy contract;
- automatic fallback or promotion to `complex<f32>`;
- support for complex values whose components use FP4, FP6, FP8, E8M0, or any
  future floating-point type not listed by this RFC;
- performance claims.

These concerns can be proposed and reviewed independently after the StableHLO
type and compatibility contract is established.

## Alternatives considered

### Support only `complex<bf16>`

This is the smallest change for bf16-specific use cases, but it does not address
the f16 request in #1794 and introduces an asymmetric exception between the two
widely used 16-bit floating-point formats. The additional StableHLO and VHLO
work required to support f16 and bf16 together is substantially shared, so this
RFC does not recommend a bf16-only type domain.

### Support `Complex<HLO_Float>`

This is mechanically concise but would include every floating-point type in
`HLO_Float`, including specialized FP4, FP6, FP8, and E8M0 formats. It would
also cause future additions to `HLO_Float` to expand the StableHLO complex type
domain without a separate compatibility decision. This RFC instead proposes a
closed component set.

### Support low-precision complex types only on FFT

This narrows the immediate verifier change but produces a fragmented type
system in which FFT can create a complex value that basic complex construction,
extraction, arithmetic, conversion, or structural operations may reject. If a
strictly FFT-only feature is desired, it should be proposed as an explicit
operation-specific type extension rather than as general StableHLO support for
low-precision complex types.

### Reuse `ComplexV1`, with or without a component-sensitive version gate

Reusing `ComplexV1` without a gate would allow new programs to appear
serializable to StableHLO versions that never specified these component
combinations. Adding a component-sensitive minimum-version gate would prevent
that particular serialization error, but it would still retroactively expand
the semantic domain of an existing VHLO type. That conflicts with VHLO's
add-only, versioned-type model and makes the meaning of `ComplexV1` depend on a
side condition outside the type version itself.

Both variants are rejected. If review establishes that VHLO must use a
different representation strategy, this RFC must be revised and approved with
the replacement compatibility contract before implementation; reuse of
`ComplexV1` is not left as an implementation-time choice.

The V1-to-V2 validation required by this proposal is not reuse of V1 for the
new feature. It rejects nonhistorical V1 component combinations and upgrades
only the f32/f64 combinations that were already valid portable StableHLO.

## Rollout and pull request boundaries

The proposed upstream sequence is:

1. Open an RFC-only PR under `rfcs/20260821-low-precision-complex.md`, change
   the header status from `Draft` to `In Review`, and make no implementation
   changes in that PR.
2. Notify OpenXLA Discuss after the RFC PR is open. The notification should
   link to the PR and direct technical discussion back to the PR so that review
   remains centralized.
3. Obtain final maintainer approval and merge the RFC before sending enabling
   implementation changes.
4. Submit one compatibility-atomic StableHLO implementation merge unit
   containing:

   - specification updates and the complete affected-surface audit;
   - ODS, verifier, and type-inference changes;
   - the VHLO version boundary and recursive type conversions;
   - positive, negative, round-trip, and compatibility tests.

   This may be one PR when reviewable, or a short stack of PRs when maintainers
   prefer smaller reviews. If stacked, the final enabling change must not land
   until every required compatibility change and test is ready; no released
   state may accept the new StableHLO types without the corresponding VHLO
   boundary.
5. If not included in step 4, submit a separate StableHLO
   reference-interpreter PR for low-precision complex tensor storage,
   constants, and numerical tests.
6. Discuss XLA, framework, legalization, and backend support in separate
   repositories and pull requests.

The RFC PR must not close #1794 by itself because approval of a design does not
implement the requested functionality. The implementation PR can close the
issue once the agreed StableHLO support is present.
