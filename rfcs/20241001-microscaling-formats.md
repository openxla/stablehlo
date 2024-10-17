# [RFC] Microscaling data types (f4E2M1FN, f6E2M3FN, f6E3M2FN, f8E8M0FNU)

Status: Review<br/>
Initial version: 10/01/2024<br/>
Last updated: 10/07/2024<br/>
Discussion thread: [PR#2581](https://github.com/openxla/stablehlo/pull/2581)

## Overview

This RFC proposes adding new primitive data types, which will allow representing
block scaled data as described in _OCP Microscaling Formats (MX) Specification
v1.0_[^1].

Dot operation on MXFP8/MXFP6/MXFP4 inputs can be hardware-accelerated on
Blackwell (NVidia architecture).

The proposed primitive types are added to LLVM APFloat[^2] [^3] [^4], MLIR
builtin types[^5] [^6] [^7] [^8], and to JAX-ML dtypes[^9] [^10], the latter
also makes them available in NumPy.

## Summary

The new types follow a convention similar to IEEE 754, but with some significant
differences.

The sub-byte data types (f4E2M1FN, f6E2M3FN, f6E3M2FN) cannot represent
infinity or NaN.

The scaling data type (f8E8M0FNU) has no mantissa and no sign bit, and it cannot
represent infinity, negative numbers or zeros.

### Float4E2M1FN

4-bit floating point type with 1 sign bit, 2 bits exponent and 1 bit mantissa.

```c
f4E2M1FN
- Exponent bias: 1
- Minimum stored exponent value: 1 (binary 01)
- Maximum stored exponent value: 3 (binary 11)
- Minimum unbiased exponent value: 1 − 1 = 0
- Maximum unbiased exponent value: 3 - 1 = 2
- Precision specifies the total number of bits used for the significand
    (mantissa), including implicit leading integer bit = 1 + 1 = 2
- Has positive and negative zero
- Doesn't have infinity
- Doesn't have NaN

Additional details:
- Zeros (+/-): S.00.0
- Max normal number (+/-): S.11.1 = 2^2 x (1+0.5) = ±6
- Min normal number (+/-): S.01.0 = 2^0 x (1+0) = ±1
- Min subnormal number (+/-): S.00.1 = 2^0 x 0.5 = ±0.5
```

List of representable absolute values: [0, 0.5, 1, 1.5, 2, 3, 4, 6]

When converting an IEEE-754 floating-point number to `f4E2M1FN`, infinities are
clamped to the minimum (-6) or maximum (6) value, depending on the sign.
Conversion from NaN is undefined.

### Float6E2M3FN

6-bit floating point type with 1 sign bit, 2 bits exponent and 3 bits mantissa.

```c
f6E2M3FN
- Exponent bias: 1
- Minimum stored exponent value: 1 (binary 01)
- Maximum stored exponent value: 3 (binary 11)
- Minimum unbiased exponent value: 1 − 1 = 0
- Maximum unbiased exponent value: 3 - 1 = 2
- Precision specifies the total number of bits used for the significand
    (mantissa), including implicit leading integer bit = 3 + 1 = 4
- Has positive and negative zero
- Doesn't have infinity
- Doesn't have NaN

Additional details:
- Zeros (+/-): S.00.000
- Max normal number (+/-): S.11.111 = 2^2 x (1+0.875) = ±7.5
- Min normal number (+/-): S.01.000 = 2^0 x (1+0) = ±1
- Max subnormal number (+/-): S.00.111 = 2^0 x 0.875 = ±0.875
- Min subnormal number (+/-): S.00.001 = 2^0 x 0.125 = ±0.125
```

When converting an IEEE-754 floating-point number to `f6E2M3FN`, infinities are
clamped to the minimum (-7.5) or maximum (7.5) value, depending on the sign.
Conversion from NaN is undefined.

### Float6E3M2FN

6-bit floating point type with 1 sign bit, 3 bits exponent and 2 bits mantissa.

```c
f6E2M3FN
- Exponent bias: 3
- Minimum stored exponent value: 1 (binary 001)
- Maximum stored exponent value: 7 (binary 111)
- Minimum unbiased exponent value: 1 − 3 = -2
- Maximum unbiased exponent value: 7 - 3 = 4
- Precision specifies the total number of bits used for the significand
    (mantissa), including implicit leading integer bit = 2 + 1 = 3
- Has positive and negative zero
- Doesn't have infinity
- Doesn't have NaN

Additional details:
- Zeros (+/-): S.000.00
- Max normal number (+/-): S.111.11 = 2^4 x (1+0.75) = ±28
- Min normal number (+/-): S.001.00 = 2^(-2) x (1+0) = ±0.25
- Max subnormal number (+/-): S.000.11 = 2^(-2) x 0.75 = ±0.1875
- Min subnormal number (+/-): S.000.01 = 2^(-2) x 0.25 = ±0.0625
```

When converting an IEEE-754 floating-point number to `f6E3M2FN`, infinities are
clamped to the minimum (-28) or maximum (28) value, depending on the sign.
Conversion from NaN is undefined.

### Float8E8M0FNU

8-bit floating point type with no sign bit, 8 bits exponent and no mantissa.

```c
f8E8M0FNU
- Exponent bias: 127
- Minimum stored exponent value: 0 (binary 0000'0000)
- Maximum stored exponent value: 254 (binary 1111'1110)
- Minimum unbiased exponent value: 0 − 127 = -127
- Maximum unbiased exponent value: 254 - 127 = 127
- Precision specifies the total number of bits used for the significand
    (mantissa), including implicit leading integer bit = 0 + 1 = 1
- Doesn't have zeros
- Doesn't have infinity
- NaN is represented as binary 1111'1111

Additional details:
- Max normal number: 1111'1110 = 2^127
- Min normal number: 0000'0000 = 2^(-127)
```

This type is intended to represent scaling factors, so there's no point of
having the sign bit or the zero value.

Converting from a 32-bit IEEE-754 floating point number (FP32) to `f8E8M0FNU`
in RTZ (round-to-zero) mode is a simple bit shift of the exponent bits.

## Microscaling formats

Microscaling format (MX) data consists of two tensors: element data as a
low-precision data type divided into equal blocks and scale data (one scaling
factor per block).

The OCP MX specification[^1] defines several floating-point formats: `MXFP8`,
`MXFP6` and `MXFP4` (described below).

### MXFP8

Block scaled data with `f8E4M3FN` or `f8E5M2` element data type and `f8E8M0FNU`
scaling data type using block size of 32 elements.

Every block of data takes 33 bytes of memory, a 48% reduction in size compared
to FP16, and a 3% overhead compared to FP8. Mean relative error of quantizing
a normal distribution of values as MXFP8 (E4M3) is ~2.5%.

The element data types already exist in StableHLO[^11] so they're not covered in
this RFC.

### MXFP6

Block scaled data with `f6E2M3FN` or `f6E3M2FN` element data type and
`f8E8M0FNU` scaling data type using block size of 32 elements.

Every block of data takes 25 bytes of memory, a 61% reduction in size compared
to FP16, and a 22% reduction compared to FP8. Mean relative error of quantizing
a normal distribution of values as MXFP6 (E2M3) is ~5%.

### MXFP4

Block scaled data with `f4E2M1FN` element data type and `f8E8M0FNU` scaling data
type using block size of 32 elements.

Every block of data takes 17 bytes of memory, a 73% reduction in size compared
to FP16, and a 47% reduction compared to FP8. Mean relative error of quantizing
a normal distribution of values as MXFP4 is ~16%.

## References

[^1]: Open Compute Project [Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
[^2]: LLVM [PR#95392](https://github.com/llvm/llvm-project/pull/95392) [APFloat] Add APFloat support for FP4 data type
[^3]: LLVM [PR#94735](https://github.com/llvm/llvm-project/pull/94735) [APFloat] Add APFloat support for FP6 data types
[^4]: LLVM [PR#107127](https://github.com/llvm/llvm-project/pull/107127) [APFloat] Add APFloat support for E8M0 type
[^5]: LLVM [PR#108877](https://github.com/llvm/llvm-project/pull/108877) [MLIR] Add f4E2M1FN type
[^6]: LLVM [PR#107999](https://github.com/llvm/llvm-project/pull/107999) [MLIR] Add f6E2M3FN type
[^7]: LLVM [PR#105573](https://github.com/llvm/llvm-project/pull/105573) [MLIR] Add f6E3M2FN type
[^8]: LLVM [PR#111028](https://github.com/llvm/llvm-project/pull/111028) [MLIR] Add f8E8M0FNU type
[^9]: JAX-ML [PR#181](https://github.com/jax-ml/ml_dtypes/pull/181) Add sub-byte data types: float4_e2m1fn, float6_e2m3fn, float6_e3m2fn
[^10]: JAX-ML [PR#166](https://github.com/jax-ml/ml_dtypes/pull/181) Add float8_e8m0_fnu (E8M0) OCP MX scale format
[^11]: [RFC: FP8 in StableHLO](https://github.com/openxla/stablehlo/blob/main/rfcs/20221031-fp8.md)
