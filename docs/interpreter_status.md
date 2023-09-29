# StableHLO Interpreter

The main goal of the StableHLO interpreter is to provide a reference
implementation to the semantics of StableHLO opset according to its
specification. The secondary goal is for the implementation to closely follow
the spec to provide additional clarity to the semantics of even the most
involved operations like `Convolution`, `Gather`/`Scatter`, and `DotGeneral`.

At the moment, OpenXLA supports the interpretation of 91 out of 96 specced
StableHLO ops, with the remaining 5 ops (`FftOp`, `RngOp`, `RngBitGeneratorOp`,
`UniformDequantizeOp`, `UniformQuantizeOp`) being a work in progress (see
[status.md](https://github.com/openxla/stablehlo/blob/main/docs/status.md) for
a complete list of ops and its latest status).

## Scope

We categorized the StableHLO opset into 11 categories consisting of 118 ops in
total (see [Appendix](#appendix)). According to our
[roadmap](https://github.com/openxla/stablehlo/blob/main/docs/roadmap.md), one
goal of StableHLO v1.0 is to implement a reference interpreter for all specced
ops from the StableHLO spec. Of the 96 ops that have a spec, we can interpret 91
ops through OpenXLA (see [Special Cases](#special-cases) for the remaining 5).

## Specification

The main requirement for the interpreter is to have 1:1 correspondence with the
spec. The spec allows standardization of the interpreter across similar ops that
lead to modular, high quality implementation of the interpreter.

## Verification

<!-- markdownlint-disable line-length -->
We carefully reviewed the spec and made sure that every constraint has a
corresponding test. We also marked every test with the constraint number that it
covers (see [ops_stablehlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/ops_stablehlo.mlir)).
This helped us to remove duplicate tests, identify missing ones, and make the
test suite easy to maintain.
<!-- markdownlint-enable line-length -->

## Type Inference

<!-- markdownlint-disable line-length -->
The interpreter results are populated by indexing over the result type, so some
development went into augmenting the type inference [implementation](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/TypeInference.cpp)
to position the interpreter to have more modular code. This allowed several
complex ops like Convolution, Gather/Scatter, and ReduceWindow to be implemented
nearly identical to how it is written in the spec.
<!-- markdownlint-enable line-length -->

## Special Cases

### Miscellaneous

This category has decomposable ops whose future is unclear at the moment. There
are three specced ops in this category that the interpreter does not support at
the moment:

* `FftOp`
* `RngOp`
* `RngBitGeneratorOp`

`FftOp` is categorized as Miscellaneous, but unlike other ops in this category,
this op does not have an expander pass, and supporting this in StableHLO is a
WIP.

`RngOp` and `RngBitGeneratorOp` can be decomposed into MHLO ops, but the
decomposition introduces a `XlaRngGetAndUpdateStateOp` which is an MHLO specific
op. Supporting interpretation of these two ops is a WIP.

<!-- markdownlint-disable line-length -->
The tool to convert remaining ops in this category to StableHLO ops that the
interpreter supports resides in [hlo_expand_main.cc](https://github.com/openxla/xla/blob/main/xla/tools/hlo_expand_main.cc).
<!-- markdownlint-enable line-length -->

### Not in HLO

Apart from the specced ops, this category consists of 10 unspecced ops (as they
plan to be moved out of StableHLO), some of which have existing passes in
[mhlo](https://github.com/openxla/xla/tree/main/xla/mlir_hlo/mhlo/transforms) to
convert them to StableHLO equivalent ops. There are three ops the interpreter
does not support because there are no existing decompositions to StableHLO ops:

* `compute_reshape_shape`
* `cstr_reshapable`
* `trace`

`compute_reshape_shape` and `cstr_reshapable` ops are part of the ongoing
Dynamism work, and they are planned to be removed from StableHLO (see
[#1668](https://github.com/openxla/stablehlo/issues/1668)).

`trace` op is private to XLA and there no no users in JAX, PyTorch or TensorFlow
(see [#604](https://github.com/openxla/stablehlo/issues/604)).

<!-- markdownlint-disable line-length -->
The tool to convert remaining ops in this category to equivalent StableHLO ops
that the interpreter supports resides in [mlir-hlo-opt.cc](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/tools/mlir-hlo-opt/mlir-hlo-opt.cc).
<!-- markdownlint-enable line-length -->

### Quantization

There are two specced ops in this category that the interpreter does not support
at the moment. These ops are part of an ongoing Quantization work. Supporting
interpretation of these two ops is a WIP:

* `UniformDequantizeOp`
* `UniformQuantizeOp`

## Build and Run the Reference Interpreter

The interpreter can be built and tested via Bazel or CMake (see
[README.md](https://github.com/openxla/stablehlo/blob/main/README.md)). To run
the interpreter, we have a translate tool to interpret StableHLO programs
written in MLIR.

```bash
stablehlo-translate --interpret <path/to/program>
```

## Testing StableHLO Programs

<!-- markdownlint-disable line-length -->
We use LLVM's [lit](https://llvm.org/docs/CommandGuide/lit.html) tool to run,
and compare against expected value to test the interpreter (see [stablehlo/tests](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests)
for sample tests).
<!-- markdownlint-enable line-length -->

A custom `Check` dialect is used to compare interpreter runtime values with
expected values. The `Check` ops are embedded into the StableHLO programs which
is run via callbacks through `stablehlo.custom_call`.

## Appendix

### Convert Miscellaneous Ops

```bash
# batch_norm_grad
hlo-expand --batch_norm_grad_expander <path/to/hlo_module>

# batch_norm_inference
hlo-expand --batch_norm_inference_expander <path/to/hlo_module>

# batch_norm_training
hlo-expand --batch_norm_training_expander <path/to/hlo_module>

# cholesky
hlo-expand --cholesky_expander <path/to/hlo_module>

# constant
# Supported in StableHLO interpreter.

# fft
# WIP

# iota
# Supported in StableHLO interpreter.

# rng
# WIP

# rng_bit_generator
# WIP

# triangular_solve
hlo-expand --triangular_solve_expander <path/to/hlo_module>
```

### Convert Not In HLO Ops

```bash
# broadcast
mlir-hlo-opt -mhlo-legalize-broadcast-to-broadcast-in-dim <path/to/input>

# compute_reshape_shape
# This op will be removed from StableHLO as part of Dynamism work (see #1668).

# create_token
mlir-hlo-opt -mhlo-legalize-create-token-to-after-all <path/to/input>

# cross-replica-sum
mlir-hlo-opt -mhlo-legalize-cross-replica-sum-to-all-reduce <path/to/input>

# cstr_reshapable
# This op will be removed from StableHLO as part of Dynamism work (see #1668).

# dot
mlir-hlo-opt -mhlo-legalize-dot-to-dot-general <path/to/input>

# einsum
mlir-hlo-opt -mhlo-legalize-einsum-to-dot-general <path/to/input>

# torch_index_select
mlir-hlo-opt -mhlo-legalize-torch-index-select-to-gather <path/to/input>

# trace
# There are no current users of trace (see #604).

# unary_einsum
mlir-hlo-opt --canonicalize -mhlo-legalize-einsum-to-dot-general <path/to/input>
```

### StableHLO Ops Categories

| Categories    | Mnemonics                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Total |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 118   |
| Control Flow  | after_all, case, if, optimization_barrier, while                                                                                                                                                                                                                                                                                                                                                                                                                            | 5     |
| Data Movement | broadcast_in_dim, concatenate, dynamic_slice, dynamic_update_slice, gather, pad, reshape, reverse, scatter, slice, sort, transpose                                                                                                                                                                                                                                                                                                                                          | 12    |
| Distribution  | all_gather, all_reduce, all_to_all, collective_permute, infeed, outfeed, partition_id, recv, reduce_scatter, replica_id, send                                                                                                                                                                                                                                                                                                                                               | 11    |
| Dynamism      | dynamic_broadcast_in_dim, dynamic_conv, dynamic_gather, dynamic_iota, dynamic_pad, dynamic_reshape, get_dimension_size, real_dynamic_slice, set_dimension_size                                                                                                                                                                                                                                                                                                              | 9     |
| Elementwise   | abs, add, and, atan2, bitcast_convert, cbrt, ceil, clamp, compare, complex, convert, cosine, count_leading_zeros, divide, exponential, exponential_minus_one, floor, imag, is_finite, log, log_plus_one, logistic, map, maximum, minimum, multiply, negate, not, or, popcnt, power, real, reduce_precision, remainder, round_nearest_afz, round_nearest_even, rsqrt, select, shift_left, shift_right_arithmetic, shift_right_logical, sign, sine, sqrt, subtract, tanh, xor | 47    |
| Extensibility | custom_call, get_tuple_element, tuple                                                                                                                                                                                                                                                                                                                                                                                                                                       | 3     |
| Miscellaneous | batch_norm_grad, batch_norm_inference, batch_norm_training, cholesky, constant, fft, iota, rng, rng_bit_generator, triangular_solve                                                                                                                                                                                                                                                                                                                                         | 10    |
| Modularity    | call, func, module, return                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 4     |
| Not In HLO    | broadcast, compute_reshape_shape, create_token, cross-replica-sum, cstr_reshapable, dot, einsum, torch_index_select, trace, unary_einsum                                                                                                                                                                                                                                                                                                                                    | 10    |
| Quantization  | uniform_dequantize, uniform_quantize                                                                                                                                                                                                                                                                                                                                                                                                                                        | 2     |
| Reduction     | convolution, dot_general, reduce, reduce_window, select_and_scatter                                                                                                                                                                                                                                                                                                                                                                                                         | 5     |
