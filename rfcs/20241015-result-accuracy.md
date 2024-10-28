# [RFC] Add result accuracy to transcendental unary ops

Status: In Review<br/>
Initial version: 10/15/2024<br/>
Last updated: 10/15/2024<br/>
Discussion thread:

## Overview

This RFC proposes adding a new attribute `result_accuracy` to the following
transcendental unary ops: `exp`, `expm1`, `log`, `logp1`, `logistic`,
`recp`, `sqrt`, `rsqrt`, `cbrt`, `sin`, `cos` and `tanh`.
`result_accuracy` allows the user to choose the implementation of these ops
based on the accuracy they request. The choice of implementation is restricted
to F32 inputs.

## Background

Transcendental ops can have multiple implementations on a processing unit that
vary in accuracy and performance. By allowing users to select implementation
based on accuracy per op, we can offer more tools to weigh tradeoffs between
accuracy and performance and also ensure more consistent numerical behaviors
across different devices.

### Caveats on targets

The proposed result_accuracy attribute can only be supported for targets that
have multiple implementations of the ops. For example, for XLA-CPU, the
implementations of these ops are dependent on LLVM and each target may have a
different implementation with a different level of accuracy. Thus, further
analysis needs to be done to support this feature on CPUs. This analysis could
be performed while building the compiler, assuming the compiler will only be
used on a single type of CPU.

## Proposed Specification

### `result_accuracy`

The users can specify the worst case numerical error they can tolerate in terms
of absolute, relative and ULP (unit in last place) errors. If they don't care
about the numerical accuracy, they can also choose the implementation using
`mode`. We propose a new attribute `result_accuracy`. `result_accuracy` can be
any combination of the following numerical tolerances `atol`, `rtol`, `ulps` or
an enum of `HIGHEST`, `DEFAULT` or `TOLERANCE`. `TOLERANCE` enum is a default
placeholder value for `mode` when the numerical tolerances are used. When using
the numerical tolerances, at least one of atol, rtol or ulps should be
specified. `HIGHEST` will give the most accurate implementation of the op
and `DEFAULT` will give the fastest implementation with less accuracy.

Name   | Type                  | Constraints
------ | --------------------- | ---------------------------------
`atol` | APFloat::IEEEdouble() | `atol >= 0`
`rtol` | APFloat::IEEEdouble() | `rtol >= 0`
`ulp`  | int64_t               | `ulp >= 0`
`mode` | EnumAttr              | `HIGHEST`, `DEFAULT`, `TOLERANCE`

```text
New Attribute:
  #stablehlo.result_accuracy<atol, rtol, ulps, mode=ResultAccuracyModeAttr>

New Enum:
  ResultAccuracyModeAttr ::= DEFAULT, HIGHEST, TOLERANCE
```

The default values are set as follows:

```text
#stablehlo.result_accuracy<atol, rtol, ulps, mode=EnumAttr>

Case1: I want DEFAULT
#stablehlo.result_accuracy<atol=0, rtol=0, ulps=0, mode=DEFAULT>

Case2: I want HIGHEST
#stablehlo.result_accuracy<atol=0, rtol=0, ulps=0, mode=HIGHEST>

Case3: I want numerical tolerance X
#stablehlo.result_accuracy<atol=X, rtol=X, ulps=X, mode=TOLERANCE>

(C1) if mode != TOLERANCE: atol = rtol = ulps = 0
```

The numerical tolerances will be compared against the compiler errors according
to the following inequality:

`abs(expected(x)-actual(x)) <= max(abs(expected(x))*max(rtol, ulps*epsilon),
atol)` for all x, where `epsilon` is the machine epsilon.

The inequality will be checked against the errors of each implementation and the
one that can satisfy the constraint will be returned. If multiple
implementations satisfy the inequality, the faster implementation will be used.
If none of the implementations can meet the requested tolerance, the compiler
will return a compile time error.

### Unary Ops

The supported ops listed above will be enhanced to support this field. The
result accuracy can appear on IR as follows:

```text
stablehlo.exp %arg0, result_accuracy = <ulps = 1> : ...
stablehlo.tanh %arg0, result_accuracy = <mode HIGHEST> : ...
stablehlo.logistic %arg0, result_accuracy = <atol=1e-4, rtol=1e-6 > : ...
```
