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
    - **yes\***: in sync with  [XLA semantics](https://www.tensorflow.org/xla/operation_semantics).
    - **yes**: in sync with [StableHLO semantics](https://github.com/openxla/stablehlo/blob/main/docs/spec_draft.md).
    - **yes(need-revisit)**: implemented but need revisit for the sync with XLA or spec
    - **infeasible**: infeasible to implement by design

## Status

| StableHLO Op             | Specification | Verification |  Type Inference   |  Pretty Printing  | Interpreter |
|:-------------------------|:-------------:|:------------:|:-----------------:|:-----------------:|:-----------:|
| abs                      |      yes      |     yes*     |       yes*        |        yes        |     no      |
| add                      |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| after_all                |      no       |      no      |        no         |        yes        |     no      |
| all_gather               |      no       |     yes*     |        no         |        no         |     no      |
| all_reduce               |      no       |      no      |        no         |        no         |     no      |
| all_to_all               |      no       |     yes*     |       yes*        |        no         |     no      |
| and                      |      yes      |     yes*     |       yes*        |        yes        |     no      |
| atan2                    |      no       |     yes*     |       yes*        |        yes        |     no      |
| batch_norm_grad          |      no       |     yes*     |       yes*        |        no         |     no      |
| batch_norm_inference     |      no       |     yes*     |       yes*        |        no         |     no      |
| batch_norm_training      |      no       |     yes*     |       yes*        |        no         |     no      |
| bitcast_convert          |      no       |     yes*     |    infeasible     |        yes        |     no      |
| broadcast_in_dim         |      no       |     yes*     |    infeasible     |        no         |     no      |
| broadcast                |      no       |     yes*     |       yes*        |        no         |     no      |
| case                     |      no       |     yes*     |       yes*        |        no         |     no      |
| cbrt                     |      no       |     yes*     |       yes*        |        yes        |     no      |
| ceil                     |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| cholesky                 |      no       |     yes*     |       yes*        |        yes        |     no      |
| clamp                    |      no       |     yes*     |       yes*        |        yes        |     no      |
| count_leading_zeros      |      no       |     yes*     |       yes*        |        yes        |     no      |
| collective_permute       |      no       |     yes*     |       yes*        |        no         |     no      |
| compare                  |      no       |     yes*     |       yes*        |        yes        |     no      |
| complex                  |      no       |     yes*     |       yes*        |        yes        |     no      |
| compute_reshape_shape    |      no       |      no      |        no         |        yes        |     no      |
| concatenate              |      no       |     yes*     |       yes*        |        yes        |     no      |
| constant                 |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| convert                  |      no       |     yes*     |    infeasible     |        yes        |     no      |
| convolution              |      no       |     yes*     |        no         | yes(need-revisit) |     no      |
| cosine                   |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| create_token             |      no       |     yes*     |        no         |        yes        |     no      |
| cross-replica-sum        |      no       |      no      |       yes*        |        no         |     no      |
| cstr_reshapable          |      no       |     yes*     |        no         |        yes        |     no      |
| custom_call              |      no       |     yes*     |    infeasible     |        yes        |     no      |
| divide                   |      yes      |     yes*     |       yes*        |        yes        |     no      |
| dot                      |      no       |     yes*     | yes(need-revisit) |        yes        |     no      |
| dot_general              |      no       |     yes*     |       yes*        |        no         |     no      |
| dynamic_broadcast_in_dim |      no       |     yes*     |        no         |        no         |     no      |
| dynamic_conv             |      no       |      no      |        no         |        no         |     no      |
| dynamic_gather           |      no       |      no      | yes(need-revisit) |        no         |     no      |
| dynamic_iota             |      no       |      no      |        no         |        yes        |     no      |
| dynamic_pad              |      no       |     yes*     |        no         |        yes        |     no      |
| dynamic_reshape          |      no       |     yes*     |        no         |        yes        |     no      |
| dynamic_slice            |      no       |     yes*     |       yes*        |        no         |     no      |
| dynamic_update_slice     |      no       |     yes*     |        no         |        yes        |     no      |
| einsum                   |      no       |      no      |        no         |        no         |     no      |
| exponential              |      yes      |     yes*     |       yes*        |        yes        |     no      |
| exponential_minus_one    |      no       |     yes*     |       yes*        |        yes        |     no      |
| fft                      |      no       |     yes*     |       yes*        |        no         |     no      |
| floor                    |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| gather                   |      no       |     yes*     |       yes*        |        no         |     no      |
| get_dimension_size       |      no       |     yes*     |        no         |        yes        |     no      |
| get_tuple_element        |      no       |     yes*     | yes(need-revisit) |        yes        |     no      |
| if                       |      no       |     yes*     |       yes*        |        no         |     no      |
| imag                     |      no       |     yes*     |       yes*        |        yes        |     no      |
| infeed                   |      no       |     yes*     |        no         |        no         |     no      |
| iota                     |      no       |     yes*     |    infeasible     |        yes        |     no      |
| is_finite                |      no       |     yes*     |       yes*        |        yes        |     no      |
| log                      |      yes      |     yes*     |       yes*        |        yes        |     no      |
| log_plus_one             |      no       |     yes*     |       yes*        |        yes        |     no      |
| logistic                 |      yes      |     yes*     |       yes*        |        yes        |     no      |
| map                      |      no       |     yes*     |        no         |        no         |     no      |
| maximum                  |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| minimum                  |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| multiply                 |      no       |     yes*     |       yes*        |        yes        |     no      |
| negate                   |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| not                      |      yes      |     yes*     |       yes*        |        yes        |     no      |
| optimization_barrier     |      no       |     yes*     |        no         |        yes        |     no      |
| or                       |      yes      |     yes*     |       yes*        |        yes        |     no      |
| outfeed                  |      no       |     yes*     |        no         |        no         |     no      |
| pad                      |      no       |     yes*     |       yes*        |        no         |     no      |
| popcnt                   |      no       |     yes*     |       yes*        |        yes        |     no      |
| power                    |      no       |     yes*     |       yes*        |        yes        |     no      |
| real_dynamic_slice       |      no       |     yes*     |        no         |        yes        |     no      |
| real                     |      no       |     yes*     |       yes*        |        yes        |     no      |
| recv                     |      no       |     yes*     |        no         |        no         |     no      |
| reduce                   |      no       |     yes*     |       yes*        | yes(need-revisit) |     no      |
| reduce_precision         |      no       |     yes*     |       yes*        |        yes        |     no      |
| reduce_scatter           |      no       |     yes*     |        no         |        no         |     no      |
| reduce_window            |      no       |     yes*     |       yes*        |        no         |     no      |
| remainder                |      yes      |     yes*     |       yes*        |        yes        |     no      |
| replica_id               |      no       |     yes*     | yes(need-revisit) |        yes        |     no      |
| reshape                  |      yes      |     yes      |    infeasible     |        yes        |     yes     |
| return                   |      no       |     yes*     |        no         |        yes        |     no      |
| reverse                  |      no       |     yes*     |       yes*        |        no         |     no      |
| rng_bit_generator        |      no       |     yes*     |    infeasible     |        yes        |     no      |
| rng                      |      no       |     yes*     |       yes*        |        yes        |     no      |
| round_nearest_afz        |      no       |     yes*     |       yes*        |        yes        |     no      |
| round_nearest_even       |      no       |     yes*     |       yes*        |        yes        |     no      |
| rsqrt                    |      yes      |     yes*     |       yes*        |        yes        |     no      |
| scatter                  |      no       |     yes*     |        no         |        no         |     no      |
| select                   |      no       |     yes*     |       yes*        |        yes        |     no      |
| select_and_scatter       |      no       |     yes*     |        no         |        no         |     no      |
| send                     |      no       |     yes*     |        no         |        no         |     no      |
| set_dimension_size       |      no       |     yes*     | yes(need-revisit) |        yes        |     no      |
| shift_left               |      no       |     yes*     |       yes*        |        yes        |     no      |
| shift_right_arithmetic   |      no       |     yes*     |       yes*        |        yes        |     no      |
| shift_right_logical      |      no       |     yes*     |       yes*        |        yes        |     no      |
| sign                     |      no       |     yes*     |       yes*        |        yes        |     no      |
| sine                     |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| slice                    |      no       |     yes*     |       yes*        |        no         |     no      |
| sort                     |      no       |     yes*     |       yes*        |        no         |     no      |
| sqrt                     |      yes      |     yes*     |       yes*        |        yes        |     no      |
| subtract                 |      yes      |     yes      |        yes        |        yes        |     yes     |
| tanh                     |      yes      |     yes*     |       yes*        |        yes        |     yes     |
| torch_index_select       |      no       |      no      |        no         |        no         |     no      |
| trace                    |      no       |     yes*     |        no         |        yes        |     no      |
| transpose                |      yes      |     yes      |        yes        |        no         |     yes     |
| triangular_solve         |      no       |     yes*     |        no         |        no         |     no      |
| tuple                    |      no       |     yes*     | yes(need-revisit) |        yes        |     no      |
| unary_einsum             |      no       |      no      |        no         |        no         |     no      |
| uniform_dequantize       |      no       |     yes*     |       yes*        |        yes        |     no      |
| uniform_quantize         |      no       |     yes*     |    infeasible     |        yes        |     no      |
| while                    |      no       |     yes*     |       yes*        | yes(need-revisit) |     no      |
| xor                      |      yes      |     yes*     |       yes*        |        yes        |     no      |
