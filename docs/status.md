## About

When bootstrapping StableHLO from MHLO, we have inherited MHLO's implementation
of many things, including prettyprinting, verification and shape inference.
Thanks to that, we already have significant coverage of the opset, but there's
still plenty to do to review the existing implementations for completeness and
provide new implementations where none exist.

This live document is for the developers and the users to track the progress on
various aspects of the opset - specification, verification, type inference,
pretty printing, interpreter, etc.

### How to use it

The progress of a StableHLO op, as mentioned in the corresponding row, on a
particular aspect, as mentioned in the corresponding column, is tracked using
one of the following tracking labels.

 - Generic labels
    - **yes**: complete
    - **no**: not complete yet, but part of [the roadmap](https://github.com/openxla/stablehlo#roadmap).
 - Customized labels for Verifier and Type Inference
    - **yes**: in sync with [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md).
    - **yes\***: in sync with [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
    - **revisit**: there is an implementation in between
      [the ODS](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td)
      and [StablehloOps.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.cpp)
      but it needs to be revisited to determine its status.
    - **infeasible**: infeasible to implement by design.

## Status

| StableHLO Op             | Specification | Verification | Type Inference | Pretty Printing | Interpreter |
|:-------------------------|:--------------|:-------------|:---------------|:----------------|:------------|
| abs                      | yes           | yes          | yes            | yes             | no          |
| add                      | yes           | yes          | yes            | yes             | yes         |
| after_all                | no            | revisit      | no             | yes             | no          |
| all_gather               | no            | revisit      | no             | no              | no          |
| all_reduce               | no            | revisit      | revisit        | no              | no          |
| all_to_all               | no            | yes*         | yes*           | no              | no          |
| and                      | yes           | yes          | yes            | yes             | yes         |
| atan2                    | no            | yes*         | yes*           | yes             | no          |
| batch_norm_grad          | no            | yes*         | yes*           | no              | no          |
| batch_norm_inference     | no            | yes*         | yes*           | no              | no          |
| batch_norm_training      | no            | yes*         | yes*           | no              | no          |
| bitcast_convert          | no            | yes*         | infeasible     | yes             | no          |
| broadcast                | no            | yes*         | yes*           | yes             | no          |
| broadcast_in_dim         | yes           | yes          | infeasible     | yes             | no          |
| case                     | yes           | revisit      | yes            | no              | no          |
| cbrt                     | no            | yes*         | yes*           | yes             | no          |
| ceil                     | yes           | yes          | yes            | yes             | yes         |
| cholesky                 | no            | yes*         | yes*           | yes             | no          |
| clamp                    | no            | yes*         | yes*           | yes             | no          |
| collective_permute       | no            | revisit      | revisit        | no              | no          |
| compare                  | no            | yes*         | yes*           | yes             | no          |
| complex                  | no            | yes*         | yes*           | yes             | no          |
| compute_reshape_shape    | no            | revisit      | no             | yes             | no          |
| concatenate              | yes           | yes          | yes            | yes             | no          |
| constant                 | yes           | yes          | yes            | yes             | yes         |
| convert                  | no            | yes*         | infeasible     | yes             | no          |
| convolution              | no            | revisit      | no             | revisit         | no          |
| cosine                   | yes           | yes          | yes            | yes             | yes         |
| count_leading_zeros      | no            | yes*         | yes*           | yes             | no          |
| create_token             | no            | revisit      | no             | yes             | no          |
| cross-replica-sum        | no            | revisit      | revisit        | no              | no          |
| cstr_reshapable          | no            | revisit      | no             | yes             | no          |
| custom_call              | no            | revisit      | infeasible     | yes             | no          |
| divide                   | yes           | yes          | yes            | yes             | no          |
| dot                      | no            | revisit      | revisit        | yes             | no          |
| dot_general              | no            | yes*         | yes*           | no              | no          |
| dynamic_broadcast_in_dim | no            | revisit      | infeasible     | no              | no          |
| dynamic_conv             | no            | revisit      | no             | no              | no          |
| dynamic_gather           | no            | revisit      | revisit        | no              | no          |
| dynamic_iota             | no            | revisit      | infeasible     | yes             | no          |
| dynamic_pad              | no            | revisit      | no             | yes             | no          |
| dynamic_reshape          | no            | revisit      | infeasible     | yes             | no          |
| dynamic_slice            | no            | yes*         | yes*           | yes             | no          |
| dynamic_update_slice     | no            | revisit      | no             | yes             | no          |
| einsum                   | no            | revisit      | no             | no              | no          |
| exponential              | yes           | yes          | yes            | yes             | no          |
| exponential_minus_one    | no            | yes*         | yes*           | yes             | no          |
| fft                      | no            | yes*         | yes*           | yes             | no          |
| floor                    | yes           | yes          | yes            | yes             | yes         |
| gather                   | no            | yes*         | yes*           | no              | no          |
| get_dimension_size       | no            | revisit      | no             | yes             | no          |
| get_tuple_element        | no            | revisit      | revisit        | yes             | no          |
| if                       | yes           | revisit      | yes            | no              | no          |
| imag                     | no            | yes*         | yes*           | yes             | no          |
| infeed                   | no            | revisit      | no             | no              | no          |
| iota                     | yes           | yes          | infeasible     | yes             | yes         |
| is_finite                | no            | yes*         | yes*           | yes             | no          |
| log                      | yes           | yes          | yes            | yes             | no          |
| log_plus_one             | no            | yes*         | yes*           | yes             | no          |
| logistic                 | yes           | yes          | yes            | yes             | no          |
| map                      | no            | yes*         | yes*           | no              | no          |
| maximum                  | yes           | yes          | yes            | yes             | yes         |
| minimum                  | yes           | yes          | yes            | yes             | yes         |
| multiply                 | yes           | yes          | yes            | yes             | yes         |
| negate                   | yes           | yes          | yes            | yes             | yes         |
| not                      | yes           | yes          | yes            | yes             | yes         |
| optimization_barrier     | no            | revisit      | no             | yes             | no          |
| or                       | yes           | yes          | yes            | yes             | yes         |
| outfeed                  | no            | revisit      | no             | no              | no          |
| pad                      | yes           | yes          | yes            | yes             | no          |
| popcnt                   | no            | yes*         | yes*           | yes             | no          |
| power                    | no            | yes*         | yes*           | yes             | no          |
| real                     | no            | yes*         | yes*           | yes             | no          |
| real_dynamic_slice       | no            | revisit      | no             | yes             | no          |
| recv                     | no            | revisit      | no             | no              | no          |
| reduce                   | no            | yes*         | yes*           | revisit         | no          |
| reduce_precision         | no            | yes*         | yes*           | yes             | no          |
| reduce_scatter           | no            | revisit      | no             | no              | no          |
| reduce_window            | no            | yes*         | yes*           | no              | no          |
| remainder                | yes           | yes          | yes            | yes             | no          |
| replica_id               | no            | revisit      | revisit        | yes             | no          |
| reshape                  | yes           | yes          | infeasible     | yes             | yes         |
| return                   | no            | revisit      | no             | yes             | no          |
| reverse                  | yes           | revisit      | yes            | yes             | no          |
| rng                      | no            | yes*         | yes*           | yes             | no          |
| rng_bit_generator        | no            | yes*         | infeasible     | yes             | no          |
| round_nearest_afz        | no            | yes*         | yes*           | yes             | no          |
| round_nearest_even       | no            | revisit      | revisit        | yes             | no          |
| rsqrt                    | yes           | yes          | yes            | yes             | no          |
| scatter                  | no            | revisit      | no             | no              | no          |
| select                   | no            | yes*         | yes*           | yes             | no          |
| select_and_scatter       | no            | revisit      | no             | no              | no          |
| send                     | no            | revisit      | no             | no              | no          |
| set_dimension_size       | no            | yes*         | yes*           | yes             | no          |
| shift_left               | no            | yes*         | yes*           | yes             | no          |
| shift_right_arithmetic   | no            | yes*         | yes*           | yes             | no          |
| shift_right_logical      | no            | yes*         | yes*           | yes             | no          |
| sign                     | no            | yes*         | yes*           | yes             | no          |
| sine                     | yes           | yes          | yes            | yes             | yes         |
| slice                    | yes           | yes          | yes            | no              | no          |
| sort                     | yes           | yes          | yes            | no              | no          |
| sqrt                     | yes           | yes          | yes            | yes             | no          |
| subtract                 | yes           | yes          | yes            | yes             | yes         |
| tanh                     | yes           | yes          | yes            | yes             | yes         |
| torch_index_select       | no            | revisit      | no             | no              | no          |
| trace                    | no            | revisit      | no             | yes             | no          |
| transpose                | yes           | yes          | yes            | yes             | yes         |
| triangular_solve         | no            | yes*         | yes*           | no              | no          |
| tuple                    | no            | revisit      | revisit        | yes             | no          |
| unary_einsum             | no            | revisit      | no             | no              | no          |
| uniform_dequantize       | no            | yes*         | yes*           | yes             | no          |
| uniform_quantize         | no            | yes*         | infeasible     | yes             | no          |
| while                    | yes           | revisit      | yes            | revisit         | no          |
| xor                      | yes           | yes          | yes            | yes             | yes         |
