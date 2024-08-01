# StableHLO status

When bootstrapping StableHLO from MHLO, we have inherited MHLO's implementation
of many things, including prettyprinting, verification and shape inference.
Thanks to that, we already have significant coverage of the opset, but there's
still plenty to do to review the existing implementations for completeness and
provide new implementations where none exist.

This live document is for the developers and the users to track the progress on
various aspects of the opset - specification, verification, type inference,
pretty printing, interpreter, etc.

## How to use it

The progress of a StableHLO op, as mentioned in the corresponding row, on a
particular aspect, as mentioned in the corresponding column, is tracked using
one of the following tracking labels.

- Generic labels
  - **yes**: there is a comprehensive implementation.
  - **no**: there is no implementation, but working on that is part of
    [the roadmap](https://github.com/openxla/stablehlo/blob/main/docs/roadmap.md).
    Note that Verifier can never be labeled as "no" because the ODS already
    implements some verification.
- Customized labels for Verifier and Type Inference
  - **yes**: there is an implementation, and it's in sync with
    [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec.md).
  - **yes\***: there is an implementation, and it's in sync with
    [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
    Since XLA semantics is oftentimes underdocumented, we are using
    [hlo_verifier.cc](https://github.com/openxla/xla/blob/main/xla/service/hlo_verifier.cc)
    and [shape_inference.cc](https://github.com/openxla/xla/blob/main/xla/service/shape_inference.cc)
    as the reference.
  - **revisit**: there is an implementation, but it doesn't fall under "yes"
    or "yes\*" - either because we haven't audited it yet, or because we have
    and found issues.
  - **infeasible**: there is no implementation, because it's infeasible.
    For example, because the result type of an op cannot be inferred from
    its operands and attributes.

## Status

| StableHLO Op             | Specification | Verification | Type Inference | Pretty Printing | Interpreter |
|:-------------------------|:--------------|:-------------|:---------------|:----------------|:------------|
| abs                      | yes           | yes          | yes            | yes             | yes         |
| add                      | yes           | yes          | yes            | yes             | yes         |
| after_all                | yes           | yes          | yes            | yes             | yes         |
| all_gather               | yes           | revisit      | no             | no              | yes         |
| all_reduce               | yes           | revisit      | yes            | no              | yes         |
| all_to_all               | yes           | revisit      | yes            | no              | yes         |
| and                      | yes           | yes          | yes            | yes             | yes         |
| atan2                    | yes           | yes          | yes            | yes             | yes         |
| batch_norm_grad          | yes           | revisit      | yes            | no              | revisit     |
| batch_norm_inference     | yes           | revisit      | yes            | no              | revisit     |
| batch_norm_training      | yes           | revisit      | yes            | no              | revisit     |
| bitcast_convert          | yes           | yes          | infeasible     | yes             | yes         |
| broadcast                | no            | yes\*        | yes\*          | yes             | revisit     |
| broadcast_in_dim         | yes           | yes          | infeasible     | yes             | yes         |
| case                     | yes           | revisit      | yes            | no              | yes         |
| cbrt                     | yes           | yes          | yes            | yes             | yes         |
| ceil                     | yes           | yes          | yes            | yes             | yes         |
| cholesky                 | yes           | yes          | yes            | yes             | revisit     |
| clamp                    | yes           | revisit      | yes            | yes             | yes         |
| collective_broadcast     | yes           | revisit      | yes            | no              | yes         |
| collective_permute       | yes           | revisit      | yes            | no              | yes         |
| compare                  | yes           | yes          | yes            | yes             | yes         |
| complex                  | yes           | yes          | yes            | yes             | yes         |
| composite                | yes           | yes          | infeasible     | yes             | yes         |
| concatenate              | yes           | yes          | yes            | yes             | yes         |
| constant                 | yes           | yes          | yes            | yes             | yes         |
| convert                  | yes           | yes          | infeasible     | yes             | yes         |
| convolution              | yes           | yes          | infeasible     | revisit         | yes         |
| cosine                   | yes           | yes          | yes            | yes             | yes         |
| count_leading_zeros      | yes           | yes          | yes            | yes             | yes         |
| create_token             | no            | yes\*        | yes\*          | yes             | revisit     |
| cross-replica-sum        | no            | revisit      | yes\*          | no              | revisit     |
| custom_call              | yes           | yes          | infeasible     | yes             | yes         |
| divide                   | yes           | yes          | yes            | yes             | yes         |
| dot                      | no            | revisit      | infeasible     | yes             | revisit     |
| dot_general              | yes           | revisit      | infeasible     | no              | yes         |
| dynamic_broadcast_in_dim | yes           | yes          | infeasible     | yes             | revisit     |
| dynamic_conv             | yes           | yes          | infeasible     | revisit         | revisit     |
| dynamic_gather           | yes           | yes          | infeasible     | no              | revisit     |
| dynamic_iota             | yes           | yes          | infeasible     | yes             | revisit     |
| dynamic_pad              | yes           | yes          | infeasible     | yes             | revisit     |
| dynamic_reshape          | yes           | yes          | infeasible     | yes             | revisit     |
| dynamic_slice            | yes           | yes          | yes            | yes             | yes         |
| dynamic_update_slice     | yes           | yes          | yes            | yes             | yes         |
| einsum                   | no            | revisit      | no             | yes             | revisit     |
| exponential              | yes           | yes          | yes            | yes             | yes         |
| exponential_minus_one    | yes           | yes          | yes            | yes             | yes         |
| fft                      | yes           | revisit      | yes            | yes             | no          |
| floor                    | yes           | yes          | yes            | yes             | yes         |
| gather                   | yes           | yes          | yes            | no              | yes         |
| get_dimension_size       | yes           | yes          | yes            | yes             | yes         |
| get_tuple_element        | yes           | yes          | yes            | yes             | yes         |
| if                       | yes           | revisit      | yes            | no              | yes         |
| imag                     | yes           | yes          | yes            | yes             | yes         |
| infeed                   | yes           | yes          | infeasible     | no              | yes         |
| iota                     | yes           | yes          | infeasible     | yes             | yes         |
| is_finite                | yes           | yes          | yes            | yes             | yes         |
| log                      | yes           | yes          | yes            | yes             | yes         |
| log_plus_one             | yes           | yes          | yes            | yes             | yes         |
| logistic                 | yes           | yes          | yes            | yes             | yes         |
| map                      | yes           | revisit      | yes            | no              | yes         |
| maximum                  | yes           | yes          | yes            | yes             | yes         |
| minimum                  | yes           | yes          | yes            | yes             | yes         |
| multiply                 | yes           | yes          | yes            | yes             | yes         |
| negate                   | yes           | yes          | yes            | yes             | yes         |
| not                      | yes           | yes          | yes            | yes             | yes         |
| optimization_barrier     | yes           | yes          | yes            | yes             | yes         |
| or                       | yes           | yes          | yes            | yes             | yes         |
| outfeed                  | yes           | yes          | yes            | no              | yes         |
| pad                      | yes           | yes          | yes            | yes             | yes         |
| partition_id             | yes           | yes          | yes            | yes             | yes         |
| popcnt                   | yes           | yes          | yes            | yes             | yes         |
| power                    | yes           | yes          | yes            | yes             | yes         |
| real                     | yes           | yes          | yes            | yes             | yes         |
| real_dynamic_slice       | no            | revisit      | no             | yes             | no          |
| recv                     | yes           | yes          | infeasible     | no              | yes         |
| reduce                   | yes           | revisit      | yes            | revisit         | yes         |
| reduce_precision         | yes           | yes          | yes            | yes             | yes         |
| reduce_scatter           | yes           | revisit      | no             | no              | yes         |
| reduce_window            | yes           | revisit      | yes            | no              | yes         |
| remainder                | yes           | yes          | yes            | yes             | yes         |
| replica_id               | yes           | yes          | yes            | yes             | yes         |
| reshape                  | yes           | yes          | infeasible     | yes             | yes         |
| return                   | no            | revisit      | infeasible     | yes             | yes         |
| reverse                  | yes           | yes          | yes            | yes             | yes         |
| rng                      | yes           | yes          | yes            | yes             | revisit     |
| rng_bit_generator        | yes           | revisit      | infeasible     | yes             | revisit     |
| round_nearest_afz        | yes           | yes          | yes            | yes             | yes         |
| round_nearest_even       | yes           | yes          | yes            | yes             | yes         |
| rsqrt                    | yes           | yes          | yes            | yes             | yes         |
| scatter                  | yes           | revisit      | yes            | no              | yes         |
| select                   | yes           | yes          | yes            | yes             | yes         |
| select_and_scatter       | yes           | revisit      | yes            | no              | yes         |
| send                     | yes           | yes          | yes            | no              | yes         |
| set_dimension_size       | no            | yes\*        | yes\*          | yes             | no          |
| shift_left               | yes           | yes          | yes            | yes             | yes         |
| shift_right_arithmetic   | yes           | yes          | yes            | yes             | yes         |
| shift_right_logical      | yes           | yes          | yes            | yes             | yes         |
| sign                     | yes           | yes          | yes            | yes             | yes         |
| sine                     | yes           | yes          | yes            | yes             | yes         |
| slice                    | yes           | yes          | yes            | no              | yes         |
| sort                     | yes           | yes          | yes            | no              | yes         |
| sqrt                     | yes           | yes          | yes            | yes             | yes         |
| subtract                 | yes           | yes          | yes            | yes             | yes         |
| tan                      | yes           | yes          | yes            | yes             | yes         |
| tanh                     | yes           | yes          | yes            | yes             | yes         |
| torch_index_select       | no            | revisit      | no             | no              | revisit     |
| transpose                | yes           | yes          | yes            | yes             | yes         |
| triangular_solve         | yes           | revisit      | yes            | no              | revisit     |
| tuple                    | yes           | yes          | yes            | yes             | yes         |
| unary_einsum             | no            | revisit      | no             | yes             | revisit     |
| uniform_dequantize       | yes           | yes          | yes            | yes             | yes         |
| uniform_quantize         | yes           | revisit      | infeasible     | yes             | yes         |
| while                    | yes           | revisit      | yes            | revisit         | yes         |
| xor                      | yes           | yes          | yes            | yes             | yes         |

## Type inference for quantized operations

The `Type Inference` column from the table above is intended to focus on
non-quantized operations. For the majority of the quantized operations, it is
not feasible to infer the result type because the quantization parameters of
the result types may vary from those of the operands. With the exception of
few cases where, operand and result types must match identically, or the op
has constraints useful to infer result type, such ops are listed below:
`all_gather`, `all_to_all`, `case`, `collective_permute`,
`compare`, `concatenate`, `constant`, `dynamic_slice`,
`dynamic_update_slice`, `gather`, `get_tuple_element`, `if`, `infeed`,
`is_finite`, `map`, `optimization_barrier`, `outfeed`, `pad`, `recv`, `reduce`,
`reduce_scatter`, `reduce_window`, `reverse`, `scatter`, `select_and_scatter`,
`send`, `slice`, `sort`, `transpose`, `tuple`, `uniform_dequantized`, `while`.
