# [RFC] Support f16- and bf16-based complex types in StableHLO

Status: In Review<br/>
Initial version: 08/21/2026<br/>
Last updated: 09/28/2026<br/>
Discussion thread: [openxla/stablehlo#2993](https://github.com/openxla/stablehlo/pull/2993)

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

This RFC defines the StableHLO and VHLO type and serialization contract for
low-precision complex types. It does not implement XLA/HLO, the
StableHLO-to-XLA bridge, runtime ABI exposure, or backend kernels. The feature
is intended to land as part of a coordinated implementation sequence with the
XLA proposal in [openxla/xla#6136](https://github.com/openxla/xla/issues/6136).
Approval of this RFC does not imply that every StableHLO consumer can execute
`complex<f16>` or `complex<bf16>`; consumers without support must reject these
types with a clear diagnostic before code generation.

### Scope and current implementation status

This RFC specifies the StableHLO/VHLO side of a cross-repository feature. The
current status of the related layers is:

| Layer | Current status | Boundary of this RFC |
| --- | --- | --- |
| StableHLO type specification | Only `complex<f32>` and `complex<f64>` are currently specified | Adds `complex<f16>` and `complex<bf16>` |
| VHLO compatibility | `ComplexV1` is the only complex representation, and its current shallow verifier does not constrain the component type | Adds a target-version-aware component-type constraint on `ComplexV1` |
| XLA/HLO core types | `C32`/`BC32` are proposed in [openxla/xla#6136](https://github.com/openxla/xla/issues/6136), but are not yet implemented | Coordinated prerequisite; not implemented here |
| StableHLO-to-XLA bridge | No `C32`/`BC32` type or constant mapping is available | Follow-up implementation |
| Runtime and backend support | [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) exposes 16-bit complex formats, but XLA does not yet expose a corresponding path | Separate runtime and backend work |

The StableHLO specification may define a valid type domain before every
consumer implements it. This RFC therefore defines the observable rejection
behavior for consumers that have not yet added execution support; it does not
claim end-to-end execution support from StableHLO serialization alone.

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
consumer. The related XLA proposal currently remains at the design stage; it
proposes adding `C32 = complex<f16>` and `BC32 = complex<bf16>` as a shared
core-type change before bridge and backend work. Consumers that do not support
the new types may reject them with a diagnostic until they add a lowering,
legalization, or native implementation. Such rejection is part of the staged
rollout, not an implicit promotion to `complex<f32>`.

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
checks whether nested types are legal for the target version, and that check
is already target-version-aware, but complex-type legality today does not
depend on the component type. Version-dependent constraint checks already
exist at the operation level: `validateConstraint` gates the reduce-family
operations and `custom_call_v1` on the target version, for example allowing
mismatched operand and result element types in reduce operations only from
0.17.0. This RFC extends that existing constraint pattern to the complex
component type rather than introducing new type-conversion infrastructure.

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

This shared-constraint change also reaches the CHLO dialect: CHLO operations
such as `broadcast_complex` and the inverse-trigonometric and hyperbolic
complex math operations use the same complex tensor predicates. The
implementation must either include those operations and their
legalization/decomposition paths in the affected-surface audit and tests, or
deliberately separate their constraints so that this RFC does not expand CHLO
acceptance accidentally.

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

StableHLO version compatibility and consumer execution capability are separate
contracts. Serialization to the feature version or a later version only
establishes that the artifact is representable by that StableHLO version. It
does not prove that an XLA, PJRT, or other consumer can lower every operation
containing the new type. A consumer that understands the StableHLO version but
lacks `C32` or `BC32` support must reject the module before code generation
with a diagnostic that identifies the unsupported type and operation.

### Proposed VHLO representation

VHLO retains `ComplexV1` as the only VHLO representation of complex types.
`ComplexV1` is structurally unchanged: one element-type parameter with the
existing shallow verifier, and an unchanged version range. The
version-dependent validity of component types is expressed as a
target-version-aware type constraint, following the pattern VHLO already uses
for operation constraints (`validateConstraint` on the reduce-family
operations and `custom_call_v1`).

The constraint is normative:

- For a target version at or after the feature version, `ComplexV1` accepts
  exactly f16-, bf16-, f32-, and f64-based components. FP8, integer, and all
  other component types are rejected.
- For a target version before the feature version, `ComplexV1` accepts only
  f32- and f64-based components. A `ComplexV1<f16>` or `ComplexV1<bf16>` value
  makes serialization to that target fail with a diagnostic, in the same way
  that a reduce operation with mismatched operand and result element types
  fails when the target version is older than 0.17.0.
- No new VHLO type or attribute is introduced, and no type-to-type conversion
  or type rewriting occurs during version conversion. Serialization of
  programs that use only f32/f64 complex types is therefore unaffected.

The implementation must add a `VHLO_VersionedTypeConstraintInterface` as the
type-level analogue of `VHLO_VersionedOpConstraintInterface`. It provides a
`validateConstraint(Type, Version)` hook. `ComplexV1` must implement the hook,
and the recursive `isLegalType` check must invoke it. A local helper may
provide the hook's implementation, but the target-aware check must have a
type-level entry point; operation-only validation is insufficient because
complex types can be nested in signatures, containers, and type-bearing
attributes. This interface is an implementation hook and does not change the
serialized VHLO type identity or its version range.

This models a component-domain relaxation as a versioned constraint rather
than as a new type version. VHLO is a serialization dialect: it is produced by
conversion from StableHLO and consumed by version conversion, not authored by
hand, and the dialect does not guarantee rejection of hand-authored payloads
that could never be produced through those flows. Every artifact is tagged
with the version it targets, and a consumer must be at least that version, so
a payload cannot be presented to an older consumer than it was serialized for.

Officially produced bytecode is validated against the requested target version
before emission. Consequently, no official pre-feature artifact can contain a
low-precision complex value. Hand-authored VHLO payloads that violate this
producer contract are outside this RFC's portable-artifact guarantees; a
consumer may reject them during parsing or conversion, and accepting one must
not establish that the pre-feature target historically supported the type.

### Recursive legality requirements

The component-type constraint must be evaluated by the recursive type
legality check with access to the requested target version. The current check
already recurses through several nested VHLO types and attributes, but the
implementation for this feature must also add explicit coverage for every
type-bearing location listed below. Checking only an operation's immediate
result types would leave invalid values hidden in containers or signatures.
The constraint must therefore be enforced at every location, including:

- `RankedTensorV1`, `UnrankedTensorV1`, and `RankedBufferV1` element types;
- `TupleV1` elements, nested tuples, and `FutureV1` element types;
- `FunctionV1` inputs and results;
- operation operands and results, function signatures, and region block
  arguments;
- `TypeV1Attr`, `TensorV1Attr`, `FloatV1Attr`, `IntegerV1Attr`, and every other
  attribute that carries a type;
- ranked-tensor encodings and nested array or dictionary attributes; and
- the storage and expressed types of `UniformQuantizedV1` and
  `UniformQuantizedPerAxisV1`, which must be traversed even when the
  quantized-type semantics reject a complex inhabitant.

If any nested occurrence fails the component-type constraint for the requested
target, the entire version conversion must fail before bytecode is written.
The diagnostic must identify the target version and the unsupported complex
component type. No path may partially convert the module or emit an artifact
containing a complex value whose component type is invalid for the target
version.

### Conversion behavior and invariants

Because no VHLO type changes, version conversion requires no type rewriting.
The observable behavior is:

1. Validate the target version as today.
2. Preserve the existing operation-version rewrites; complex types are never
   rewritten.
3. Extend the recursive type legality check to evaluate the component-type
   constraint against the requested target version over every type-bearing
   location listed above.
4. If any check fails, fail the pass before bytecode emission with a
   diagnostic naming the target version and the unsupported component type.

The transformation remains all-or-nothing from the serializer's perspective,
and serialization must not emit bytecode from a module whose complex component
types are invalid for the requested target version.

## Consumer behavior

StableHLO verifier acceptance is not a claim that every consumer can execute
the new types.

StableHLO serialization validates only StableHLO/VHLO representability. A
StableHLO-to-XLA bridge must reject `complex<f16>` or `complex<bf16>` before
constructing the XLA/HLO module when the corresponding XLA primitive type is
unavailable. The diagnostic must identify the unsupported type, operation, and
target. Other consumers must provide an equivalent pre-code-generation
capability check or a separately specified legalization path.

A consumer that lacks support may:

- reject a module containing the new types with a clear unsupported-type
  diagnostic;
- apply a separately specified legalization;
- implement the types natively.

This RFC does not prescribe which option a consumer must choose. It does
require that unsupported consumers fail explicitly before code generation. In
particular, a consumer must not reinterpret a four-byte value as `C64`, fall
through an existing `C64`/`C128` path, or silently promote the declared value to
`complex<f32>`. XLA support is not a prerequisite for StableHLO type validity,
but the StableHLO implementation and the XLA/HLO work must be coordinated
before the feature is presented as end-to-end executable.

## Verification and testing

The implementation must include positive tests for:

- StableHLO verifier acceptance and exact type inference for the complete
  specified domain, independently of whether a particular consumer executes
  every operation;
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
- deserialization of historical f32/f64 complex artifacts, which requires no
  type rewriting under this proposal;
- serialization of `complex<f16>` and `complex<bf16>` values, including values
  nested in every semantically valid type-bearing container and attribute
  listed above, to the feature version and later targets; and
- continued old-version serialization of f32/f64 complex programs.

The implementation must include negative tests for:

- consumer-side rejection before code generation when `C32` or `BC32` support
  is absent, including a diagnostic that names the unsupported type, operation,
  and target;
- complex values with integer component types;
- at least one floating-point component type outside this RFC, such as an FP8
  type;
- rejection of serialization to any target version before the feature version
  when a `ComplexV1<f16>` or `ComplexV1<bf16>` value is present, both directly
  and when nested in every semantically valid supported type-bearing container
  or attribute listed above, enforced by the version-conversion constraint
  rather than by tightening the shallow V1 parser or verifier;
- rejection of complex types placed in quantized-type storage or expressed
  type parameters by the semantic constraint owned by the corresponding
  StableHLO or consumer layer, while those locations are still traversed for
  VHLO version legality;
- a complex value with an FP8, integer, or other unsupported component, for
  any target version;
- mismatched f16/bf16 inputs to `stablehlo.complex`;
- mismatched real/complex component types in FFT input and result types;
- attempting to serialize a program containing `complex<f16>` or
  `complex<bf16>` to the version immediately before the feature version; and
- failure before bytecode emission, with a diagnostic that names both the
  requested target version and the unsupported nested type.

The VHLO compatibility suite must cover the feature version following the
existing VHLO checklist, including fixtures that exercise the component-type
constraint against old and new target versions.

Reference-interpreter numerical tests may be delivered in a separate
StableHLO-only follow-up if the required tensor storage and constant
materialization changes would obscure review of the type-system and
compatibility change. If maintainers require reference execution as an opset
acceptance criterion, that follow-up can be stacked or folded into the
implementation change without adding XLA or backend work.

## Non-goals

This RFC does not propose:

- the implementation of an XLA primitive type or StableHLO-to-XLA bridge;
  those pieces are coordinated with [openxla/xla#6136](https://github.com/openxla/xla/issues/6136)
  and are required before claiming end-to-end execution;
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

These concerns are separate implementation stages, but they are not an
indefinite dependency: the enabling StableHLO implementation must have an
agreed XLA/HLO core-type plan and an explicit unsupported-consumer path before
it lands. The bridge, runtime, and backend changes may remain separate pull
requests.

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

### Reuse `ComplexV1` with target-aware validation (chosen)

VHLO keeps `ComplexV1` as the only complex representation and expresses the
version-dependent validity of component types as a target-version-aware type
constraint, the same pattern the dialect already uses for operation
constraints. Review of this RFC directed this approach over a new type
version: VHLO is a serialization dialect produced by conversion and consumed
by version conversion rather than authored by hand, every artifact is tagged
with its target version, and a downgrade containing values invalid for the
target already fails today. Modeling a component-domain relaxation with the
constraint mechanism keeps the category of change consistent with existing
practice and requires far less code.

### Add `ComplexV2` at the feature version (declined)

A new `ComplexV2` with a closed component set keeps the meaning of a complex
type intrinsic to the type version and makes the compatibility boundary
inspectable without target context. It would, however, require the dialect's
first type version split together with new type-to-type conversion
infrastructure: recursive rewriting of enclosing types, function signatures,
and type-bearing attributes for a change that does not alter the structure of
the type. Given VHLO's serialization-dialect stance, that cost is not
justified for this category of change.

The alternatives are:

| Design | Benefit | Cost |
| --- | --- | --- |
| Target-aware validation on `ComplexV1` | Consistent with the existing versioned-constraint pattern; no type rewriting; far less code | Complex-type validity depends on the target version rather than the type alone |
| `ComplexV2` with V1/V2 conversion | Type-intrinsic meaning and an inspectable boundary without target context | First VHLO type version split plus recursive type-conversion infrastructure |

Reusing `ComplexV1` with no version check at all would allow new programs to
appear serializable to StableHLO versions that never specified these
component combinations, and remains rejected.

## Rollout and pull request boundaries

The proposed upstream sequence is deliberately split by capability:

1. Keep this RFC as a design-only PR and confirm the cross-repository contract
   with [openxla/xla#6136](https://github.com/openxla/xla/issues/6136). RFC
   approval does not itself enable serialization or execution.
2. Obtain agreement on the XLA/HLO core representation (`C32` and `BC32`),
   host storage, `LiteralProto`, parser/printer behavior, and explicit
   unsupported-execution diagnostics. The current XLA proposal places this
   core-type change before bridge and backend work.
3. After the RFC is approved, submit the compatibility-atomic StableHLO
   implementation, including:

   - specification, ODS, verifier, and type-inference changes;
   - the target-version-aware `ComplexV1` component-type constraint and its
     recursive legality coverage;
   - positive, negative, round-trip, and serialization tests;
   - tests that reject unsupported target versions before bytecode emission.

   The affected-surface audit must include CHLO operations that inherit the
   shared complex predicates, or record an explicit constraint boundary that
   excludes them.

   The implementation may be developed in parallel with the XLA core change,
   but it must not be presented as end-to-end execution support. No released
   state may emit the new StableHLO types without the corresponding VHLO
   boundary.
4. Implement the StableHLO-to-XLA bridge, Shape mapping, constant materialization,
   and bit-exact round trips in a separate follow-up. A bridge that has not yet
   added `C32`/`BC32` must reject those types before code generation.
5. Add explicit legalization policies, keeping pair decomposition and
   promotion to `C64` as separate choices. The core type change must not choose
   either policy implicitly.
6. Add runtime ABI exposure and backend implementations, beginning with a
   concrete backend path such as the cuFFT 16-bit complex formats and then
   extending support as maintainers approve.

The RFC PR must not close #1794 by itself because design approval does not
implement the requested functionality. The implementation PR can close the
issue only after the agreed StableHLO support and the required consumer path
are present.
