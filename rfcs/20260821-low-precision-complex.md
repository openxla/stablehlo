# [RFC] Support f16- and bf16-based complex types in StableHLO

Status: In Review<br/>
Initial version: 08/21/2026<br/>
Last updated: 09/15/2026<br/>
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
| VHLO compatibility | `ComplexV1` is the historical representation | Adds the feature-version `ComplexV2` boundary |
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
contracts. Serialization to a version that defines `ComplexV2` only establishes
that the artifact is representable by that StableHLO version. It does not prove
that an XLA, PJRT, or other consumer can lower every operation containing the
new type. A consumer that understands the StableHLO version but lacks `C32` or
`BC32` support must reject the module before code generation with a diagnostic
that identifies the unsupported type and operation.

### Proposed VHLO representation

VHLO currently has `ComplexV1`, available since 0.9.0. Its representation is
structurally capable of containing any VHLO component type, but historical
StableHLO versions only specified `complex<f32>` and `complex<f64>`. Reusing
`ComplexV1` for f16/bf16 without an additional version boundary would make a
new StableHLO feature appear incorrectly available to old target versions.

This RFC requires adding `ComplexV2` at the new feature version. The versioned
contract is normative:

- `ComplexV1` remains the historical VHLO representation. Its maximum supported
  version is the patch-zero version of the minor immediately preceding the
  feature version (for example, `1.20.0` when the feature version is `1.21.0`).
  The valid portable StableHLO domain represented by historical V1 artifacts is
  f32- and f64-based complex types.
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

- consumer-side rejection before code generation when `C32` or `BC32` support
  is absent, including a diagnostic that names the unsupported type, operation,
  and target;
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

### Reuse `ComplexV1` with target-aware validation

A target-aware constraint interface could reject `ComplexV1<f16>` and
`ComplexV1<bf16>` when converting to an older target. This would reduce some
conversion code, but it would not preserve the historical meaning of
`ComplexV1`: the same VHLO type would acquire a different semantic domain
based on an external target-version predicate. It would also leave
structurally representable, but historically invalid, V1 values in the parser
and require every nested type-bearing location to preserve the same side
condition.

The alternatives are:

| Design | Benefit | Compatibility cost |
| --- | --- | --- |
| Target-aware validation on `ComplexV1` | Less dedicated type conversion | Retroactively changes V1 semantics and makes validity depend on target context |
| `ComplexV2` with V1/V2 conversion | Explicit version boundary and historical V1 meaning | Requires recursive conversion and downgrade tests |

Reusing `ComplexV1` without a gate would allow new programs to appear
serializable to StableHLO versions that never specified these component
combinations. Adding a component-sensitive minimum-version gate would prevent
that particular serialization error, but it would still retroactively expand
the semantic domain of an existing VHLO type. That conflicts with VHLO's
add-only, versioned-type model and makes the meaning of `ComplexV1` depend on a
side condition outside the type version itself.

This RFC chooses `ComplexV2`. A target-aware type constraint may still be
useful as an implementation mechanism for recursive legality checks, but it is
not a replacement for the V1-to-V2 semantic boundary. `ComplexV2` keeps the
historical meaning of `ComplexV1` explicit and makes the compatibility boundary
inspectable. VHLO maintainer confirmation of this choice is required before
implementation.

The V1-to-V2 validation required by this proposal is not reuse of V1 for the
new feature. It rejects nonhistorical V1 component combinations and upgrades
only the f32/f64 combinations that were already valid portable StableHLO.

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
   - the VHLO `ComplexV2` boundary and recursive conversions;
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
