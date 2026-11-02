# [RFC] Negative value conversion for unsigned float types

Status: in review<br/>
Initial version: 2026-10-21<br/>
Last updated: 2026-10-21<br/>
Discussion thread: [PR#3050](https://github.com/openxla/stablehlo/pull/3050)

## Summary

This RFC proposes refined semantics for `convert` when the destination is an
unsigned floating-point type.

## Motivation

Support for unsigned floating-point types is relatively new; StableHLO already
supports `f8E8M0FNU`, and `f8E5M3FNU` is in progress (#3037).

Currently, `convert` is specified as follows (the wording and #180 are concerned
with rounding mode, not negative values):

> if the source value can be exactly represented in the destination type, the
> result value is that exact representation. **Otherwise, the behavior is TBD
> (#180).**

## Proposal

We propose that StableHLO `convert` should have well-defined semantics for
negative inputs when converting to an unsigned float type. Namely:

- Finite negative values (`< 0`) convert to `nan`.
- `-0.0` converts
  - to `+0.0` if zero is representable
  - to `nan` otherwise. (`f8E8M0FNU` cannot represent zero.)
- `-nan` converts to `nan`.
- `-inf` converts to `nan`.
- Rounding behavior is left unspecified, as it was.

No existing behavior is changed; `f8E8M0FNU` already implements it this way.

## Rationale

This matches the behavior implemented by `ml_dtypes`, and what is already
implemented as a special case for `f8E8M0FNU` on GPU in
`xla/codegen/emitters/transforms/expand_float_ops.cc`:

```cpp
  if (llvm::isa<mlir::Float8E8M0FNUType>(to_ty)) {
    // Converting a negative number to E8M0 results in NaN.
    input_is_nan = from_sign_bit | input_is_nan;
```

Notably NVIDIA PTX specifies a different behavior; PTX `cvt` for `e8m0` and
`ue5m3` converts the absolute value for negative inputs. As we don't think this
generalizes well, we want to make StableHLO less surprising. The lowering passes
(`EmitFloatConversion` and NVIDIA-specific lowerings) will be updated to take
care of the semantics difference.
