# [RFC] Add NUMPY to ComparisonType and Deprecate SIGNED/UNSIGNED

Status: In Review
Initial version: 2026-09-01
Last updated: 2026-09-01
Discussion thread:
[GitHub PR #3003](https://github.com/openxla/stablehlo/pull/3003)

## Overview

This RFC proposes:

1. Adding `NUMPY` (or `FLOAT_NUMPY`) to `ComparisonType` (`compare_type`
   attribute on `stablehlo.compare`).
2. Formally deprecating the `SIGNED` and `UNSIGNED` enum values in
   `ComparisonType`.
3. Updating the StableHLO specification for `compare` to support
   NumPy/Python-style total ordering semantics.

## Motivation & Background

### 1. The Need for NUMPY Comparison Order

Machine learning frontends (JAX, PyTorch, NumPy) require NumPy sorting and
ranking semantics:

- -0.0 and +0.0 are treated as equivalent keys (preserving stability in stable
  sorts).
- All NaNs (both positive and negative) are sorted to the end of the sequence
  (> +infinity).
StableHLO currently supports only `FLOAT` (standard IEEE-754 partial order)
and `TOTALORDER` (IEEE-754 totalOrder, where -NaN < -infinity and
-0.0 < +0.0).
Because of this, frontends like JAX must emit a 7+ instruction boilerplate
subgraph per scalar comparison in every sort comparator to canonicalize zeros
and NaNs before calling `TOTALORDER`. In turn, downstream compiler backends
(such as XLA GPU's `SortRewriter`) rely on fragile AST pattern matching to
detect this subgraph and emit fast CUB `DeviceRadixSort` (see b/376918731).
If any compiler optimization alters the AST, pattern matching fails and falls
back to a 10x-25x slower bitonic sort.
Adding `NUMPY` directly to `ComparisonType` eliminates frontend
canonicalization bloat and provides a clean, robust contract for compiler
backends.

### 2. Redundancy of SIGNED and UNSIGNED

In StableHLO and MLIR, operand types explicitly encode signedness
(`si8`..`si64`, `ui8`..`ui64`, `i1`). Integer comparisons are mathematically
total, and their sign interpretation is strictly dictated by the operand's
`IntegerType`.
Having `SIGNED` and `UNSIGNED` in `ComparisonType` is copied from XLA internals.
XLA is migrating away from this by separating data type from ordering. In the
future, the data type will always be derived from the operands of the
comparison. In StableHLO, `SIGNED` and `UNSIGNED` are redundant and should be
deprecated in favor of omitting `compare_type` (or using `NOTYPE`) for
integer/boolean operands.

## Proposed Changes

### 1. Dialect & Attributes

Update `StableHLO_ComparisonType`:

```tablegen
def STABLEHLO_COMPARISON_TYPE_NOTYPE : I32EnumAttrCase<"NOTYPE", 0>;
def STABLEHLO_COMPARISON_TYPE_FLOAT : I32EnumAttrCase<"FLOAT", 1>;
def STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER :
    I32EnumAttrCase<"TOTALORDER", 2>;
def STABLEHLO_COMPARISON_TYPE_SIGNED :
    I32EnumAttrCase<"SIGNED", 3>; // Deprecated
def STABLEHLO_COMPARISON_TYPE_UNSIGNED :
    I32EnumAttrCase<"UNSIGNED", 4>; // Deprecated
def STABLEHLO_COMPARISON_TYPE_NUMPY : I32EnumAttrCase<"NUMPY", 5>;
def StableHLO_ComparisonType : I32EnumAttr<"ComparisonType",
    "Which comparison type to use.",
    [
      STABLEHLO_COMPARISON_TYPE_NOTYPE,
      STABLEHLO_COMPARISON_TYPE_FLOAT,
      STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
      STABLEHLO_COMPARISON_TYPE_SIGNED,
      STABLEHLO_COMPARISON_TYPE_UNSIGNED,
      STABLEHLO_COMPARISON_TYPE_NUMPY
    ]>
```

### 2. Specification (`docs/spec.md`)

Update `compare` semantics:

- For floating-point element types with `compare_type = NUMPY`:
  - Implements total weak ordering:
    -infinity < finite < -0.0 == +0.0 < finite < +infinity < NaN.
  - -0.0 and +0.0 compare as equal (`EQ` is true; neither is `<` the other).
  - All NaN representations (positive, negative, signaling, quiet) compare as
    equal to each other and greater than all non-NaN values.
- Constraints (C3) updated:
  - `SIGNED` and `UNSIGNED` marked deprecated.
  - `NOTYPE` is the standard for integer and boolean types.
  - `FLOAT`, `TOTALORDER`, or `NUMPY` valid for floating-point types.

### 3. Compatibility & Versioning

- **Backward Compatibility**: Existing VHLO bytecodes with `SIGNED`/`UNSIGNED`
  will continue to deserialize. On upgrade, they are preserved or mapped to
  `NOTYPE`.
- **Forward Compatibility**: `NUMPY` will be added to `VhloEnums.td`
  (`VHLO_ComparisonTypeV1`) with appropriate versioning.
  
## Alternatives Considered

### Alternative 1: Replace `compare_type` with `comparison_order` attribute

Rather than extending `ComparisonType`, deprecate `compare_type` entirely and
introduce `comparison_order` with values `PARTIAL`, `TOTAL`, `NUMPY`.

- **Pros**: Cleaner conceptual model (separates data type from ordering; 1:1
  match with XLA IR's `Comparison::Order`).
- **Cons**: Substantial churn for all existing StableHLO producers and
  consumers, requiring op signature migration (`CompareOpV2` or complex
  bytecode upgrade rules). Option A achieves the required semantic
  expressiveness with zero disruption.
  
### Alternative 2: Keep frontend canonicalization

Continue lowering NumPy sorts in JAX/PyTorch via select/compare ASTs and
relying on XLA backend pattern matching.

- **Cons**: Brittle compiler pattern matching logic, risk of having lower
  performance, unnecessary IR complexity.
