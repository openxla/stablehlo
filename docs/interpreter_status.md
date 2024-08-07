# StableHLO Interpreter

The main goal of the StableHLO interpreter is to provide a reference
implementation to the semantics of StableHLO opset according to its
specification. The secondary goal is for the implementation to closely follow
the spec, favoring readability over performance, to provide additional clarity
to the semantics of even the most involved operations like `Convolution`,
`Gather`/`Scatter`, and `DotGeneral`.

At the moment, OpenXLA supports the interpretation of 91 out of 96 specced
StableHLO ops. The remaining 3 ops (`FftOp`, `RngOp`, `RngBitGeneratorOp`) have
their semantics documented in
[spec.md](https://github.com/openxla/stablehlo/blob/main/docs/spec.md), and have
completed initial investigations on how to move forward (see
[status.md](https://github.com/openxla/stablehlo/blob/main/docs/status.md)
for a complete list of ops and its latest status). These final
enhancements will be implemented on an as-needed community basis.

## Scope

We categorized the StableHLO opset into 11 categories consisting of 118 ops in
total (see [Appendix](#appendix)).
[Reference Implementation](https://github.com/orgs/openxla/projects/7)
workstream organizes the work on implementing [an interpreter](https://github.com/openxla/stablehlo/blob/main/docs/reference.md)
for 100% of StableHLO ops as defined in the StableHLO specification. We are
planning to complete all or almost all work in this workstream in StableHLO
v1.0. Of the 96 ops that have a spec currently, we can interpret 91 ops through
OpenXLA (see [Special Cases](#special-cases) for the remaining 5).

## Specification

The main requirement for the interpreter is to have 1:1 correspondence with the
spec. The spec allows standardization of the interpreter across similar ops that
lead to modular, high quality implementation of the interpreter.

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

Apart from the specced ops, this category consists of 8 unspecced ops (see
[StableHLO Ops Categories](#stablehlo-ops-categories)) which are planned to be
moved out of StableHLO. Most of these ops have existing passes in
[mhlo](https://github.com/openxla/xla/tree/main/xla/mlir_hlo/mhlo/transforms) to
convert them to StableHLO equivalent ops.

<!-- markdownlint-disable line-length -->
The tool to convert remaining ops in this category to equivalent StableHLO ops
that the interpreter supports resides in [mlir-hlo-opt.cc](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/tools/mlir-hlo-opt/mlir-hlo-opt.cc).
<!-- markdownlint-enable line-length -->

### Quantization

Interpreter support for `stablehlo.constant` operation with quantized type is
unsupported and tracked via
[#1691](https://github.com/openxla/stablehlo/issues/1691).

## Usage Instructions

### Building the Reference Interpreter

The interpreter can be built and tested via Bazel or CMake (preferred). For full
instructions, see [README.md](https://github.com/openxla/stablehlo/blob/main/README.md).

Bazel:

```sh
bazel build //...
```

CMake:

```sh
mkdir -p build && cd build

cmake .. -GNinja \
  -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
```

To run the interpreter, we have a translate tool to interpret StableHLO programs
written in MLIR.

```sh
stablehlo-translate --interpret <path/to/program>
```

### The Interpreter Dialect

<!-- markdownlint-disable line-length -->
The `Interpreter` dialect contains various utility ops related to the
interpreter. Specifically, the `interpreter.run_parallel` (see
[InterpreterOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/reference/InterpreterOps.td)
for op semantics and example usage) op allows interpretation of Distribution ops, and more
utilities plan to be added based on community needs.
<!-- markdownlint-enable line-length -->

### The Check Dialect

<!-- markdownlint-disable line-length -->
The `Check` dialect is used to compare interpreter runtime values to expected
values. StableHLO program outputs can be tested via various check ops (see
[CheckOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/CheckOps.td)
for op semantics and  example usage).
<!-- markdownlint-enable line-length -->

### Writing Test Programs

<!-- markdownlint-disable line-length -->
We use LLVM's [lit](https://llvm.org/docs/CommandGuide/lit.html) tool to run and
compare against generated file to diff against the output of the interpreter
(see [stablehlo/tests/interpret](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret)
for example tests).

Testing `AddOp` (sample from
[interpret_add.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/interpret/add.mlir)):
<!-- markdownlint-enable line-length -->

```mlir
// RUN: stablehlo-translate --interpret %s

func.func @add_op_scalar() {
  %0 = stablehlo.constant dense<2> : tensor<i4>
  %1 = stablehlo.constant dense<3> : tensor<i4>
  %2 = stablehlo.add %0, %1 : tensor<i4>
  check.expect_eq_const %2, dense<5> : tensor<i4>
  func.return
}
```

Testing ops in the Distribution category requires running it via the
`interpreter.run_parallel` utility op.

<!-- markdownlint-disable line-length -->
Testing `AllReduceOp` (sample from
[all_reduce.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/interpret/all_reduce.mlir)):
<!-- markdownlint-enable line-length -->

```mlir
// RUN: stablehlo-translate --interpret %s

module @cross_replica {
  func.func public @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
  }
  func.func public @main() {
    %inputs0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
    %inputs1 = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
    %results:2 = "interpreter.run_parallel"(%inputs0, %inputs1) {
      programs=[[@all_reduce], [@all_reduce]]
    } : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    check.expect_eq_const %results#0, dense<[6, 8, 10, 12]> : tensor<4xi64>
    check.expect_eq_const %results#1, dense<[6, 8, 10, 12]> : tensor<4xi64>
    func.return
  }
}
```

### Debugging StableHLO

Following the StableHLO build steps, the StableHLO binaries for tools in
`stablehlo/tools` should reside in `/build/bin`. Common debugging tools like
GDB can be used to step through the code:

```sh
gdb --args ./build/bin/stablehlo-translate -allow-unregistered-dialect --interpret ./stablehlo/tests/interpret/<test>.mlir
```

## Appendix

### Convert Miscellaneous Ops

```sh
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
# TBD

# iota
# Supported in StableHLO interpreter.

# rng
# TBD

# rng_bit_generator
# TBD

# triangular_solve
hlo-expand --triangular_solve_expander <path/to/hlo_module>
```

### Convert Not In HLO Ops

```sh
# broadcast
mlir-hlo-opt -mhlo-legalize-broadcast-to-broadcast-in-dim <path/to/input>

# create_token
mlir-hlo-opt -mhlo-legalize-create-token-to-after-all <path/to/input>

# cross-replica-sum
mlir-hlo-opt -mhlo-legalize-cross-replica-sum-to-all-reduce <path/to/input>

# dot
mlir-hlo-opt -mhlo-legalize-dot-to-dot-general <path/to/input>

# einsum
mlir-hlo-opt -mhlo-legalize-einsum-to-dot-general <path/to/input>

# torch_index_select
mlir-hlo-opt -mhlo-legalize-torch-index-select-to-gather <path/to/input>

# unary_einsum
mlir-hlo-opt --canonicalize -mhlo-legalize-einsum-to-dot-general <path/to/input>
```

### StableHLO Ops Categories

| Categories    | Mnemonics                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Total |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 119   |
| Control Flow  | after_all, case, if, optimization_barrier, while                                                                                                                                                                                                                                                                                                                                                                                                                            | 5     |
| Data Movement | broadcast_in_dim, concatenate, dynamic_slice, dynamic_update_slice, gather, pad, reshape, reverse, scatter, slice, sort, transpose                                                                                                                                                                                                                                                                                                                                          | 12    |
| Distribution  | all_gather, all_reduce, all_to_all, collective_permute, infeed, outfeed, partition_id, recv, reduce_scatter, replica_id, send                                                                                                                                                                                                                                                                                                                                               | 11    |
| Dynamism      | dynamic_broadcast_in_dim, dynamic_conv, dynamic_gather, dynamic_iota, dynamic_pad, dynamic_reshape, get_dimension_size, real_dynamic_slice, set_dimension_size                                                                                                                                                                                                                                                                                                              | 9     |
| Elementwise   | abs, add, and, atan2, bitcast_convert, cbrt, ceil, clamp, compare, complex, convert, cosine, count_leading_zeros, divide, exponential, exponential_minus_one, floor, imag, is_finite, log, log_plus_one, logistic, map, maximum, minimum, multiply, negate, not, or, popcnt, power, real, reduce_precision, remainder, round_nearest_afz, round_nearest_even, rsqrt, select, shift_left, shift_right_arithmetic, shift_right_logical, sign, sine, sqrt, subtract, tan, tanh, xor | 48    |
| Extensibility | custom_call, get_tuple_element, tuple                                                                                                                                                                                                                                                                                                                                                                                                                                       | 3     |
| Miscellaneous | batch_norm_grad, batch_norm_inference, batch_norm_training, cholesky, constant, fft, iota, rng, rng_bit_generator, triangular_solve                                                                                                                                                                                                                                                                                                                                         | 10    |
| Modularity    | call, func, module, return                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 4     |
| Not In HLO    | broadcast, create_token, cross-replica-sum, dot, einsum, torch_index_select, unary_einsum                                                                                                                                                                                                                                                                                                                                                                            | 8     |
| Quantization  | uniform_dequantize, uniform_quantize                                                                                                                                                                                                                                                                                                                                                                                                                                        | 2     |
| Reduction     | convolution, dot_general, reduce, reduce_window, select_and_scatter                                                                                                                                                                                                                                                                                                                                                                                                         | 5     |
