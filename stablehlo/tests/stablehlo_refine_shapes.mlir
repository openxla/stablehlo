// RUN: stablehlo-opt --stablehlo-refine-shapes --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @error_illformed(%arg0: tensor<3xf32>, %arg1: tensor<4xf32>) -> tensor<?xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<3xf32>) -> tensor<?xf32>
  %1 = stablehlo.abs %arg1 : (tensor<4xf32>) -> tensor<?xf32>
  // expected-error@+1{{requires compatible types for all operands and results}}
  %2 = stablehlo.add %0, %1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// -----

// expected-error@+1{{must have exactly one block}}
func.func @error_too_many_blocks(%arg0: tensor<f32>) -> tensor<f32> {
  cf.br ^bb1(%arg0 : tensor<f32>)
^bb1(%arg1 : tensor<f32>):
  func.return %arg1 : tensor<f32>
}

// -----

// expected-error@-3{{must have no more than one function}}
func.func @error_too_many_functions(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = func.call @helper(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

func.func private @helper(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

func.func @error_unsupported_operation(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> index {
  // CHECK: stablehlo.add{{.*}} -> tensor<?xf32>
  %0 = stablehlo.add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<?xf32>
  %1 = arith.constant 0 : index
  %2 = tensor.dim %0, %1 : tensor<?xf32>
  func.return %2 : index
}

// -----

// CHECK-LABEL: func @eval_add
func.func @eval_add() -> tensor<i64> {
  // CHECK-NOT: stablehlo.add
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<4> : tensor<i64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.add %0, %1 : tensor<i64>
  func.return %2 : tensor<i64>
}

// -----

// CHECK-LABEL: func @eval_and
func.func @eval_and() -> tensor<i1> {
  // CHECK-NOT: stablehlo.and
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<true> : tensor<i1>
  %1 = stablehlo.constant dense<true> : tensor<i1>
  %2 = stablehlo.and %0, %1 : tensor<i1>
  func.return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func @eval_broadcast_in_dim
func.func @eval_broadcast_in_dim() -> tensor<1xi64> {
  // CHECK-NOT: stablehlo.broadcast_in_dim
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<1> : tensor<1xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1xi64>
  func.return %1 : tensor<1xi64>
}

// -----

// CHECK-LABEL: func @eval_compare_eq
func.func @eval_compare_eq() -> tensor<i1> {
  // CHECK-NOT: stablehlo.compare
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.compare EQ, %0, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func @eval_compare_ne
func.func @eval_compare_ne() -> tensor<i1> {
  // CHECK-NOT: stablehlo.compare
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.compare NE, %0, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func @eval_compare_ge
func.func @eval_compare_ge() -> tensor<i1> {
  // CHECK-NOT: stablehlo.compare
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.compare GE, %0, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func @eval_compare_gt
func.func @eval_compare_gt() -> tensor<i1> {
  // CHECK-NOT: stablehlo.compare
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.compare GT, %0, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func @eval_compare_le
func.func @eval_compare_le() -> tensor<i1> {
  // CHECK-NOT: stablehlo.compare
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.compare LE, %0, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func @eval_compare_lt
func.func @eval_compare_lt() -> tensor<i1> {
  // CHECK-NOT: stablehlo.compare
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.compare LT, %0, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: func @eval_concatenate_1d
func.func @eval_concatenate_1d() -> tensor<4xi64> {
  // CHECK-NOT: stablehlo.concatenate
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<2xi64>, tensor<2xi64>) -> tensor<4xi64>
  func.return %2 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @eval_concatenate_2d
func.func @eval_concatenate_2d() -> tensor<2x2xi64> {
  // CHECK-NOT: stablehlo.concatenate
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<{{\[}}[1, 2], [3, 4]]> : tensor<2x2xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi64>
  %1 = stablehlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<2x2xi64>
  func.return %2 : tensor<2x2xi64>
}

// -----

// CHECK-LABEL: func @eval_convert
func.func @eval_convert() -> tensor<i64> {
  // CHECK-NOT: stablehlo.convert
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<4> : tensor<i64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<4> : tensor<i32>
  %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
  func.return %1 : tensor<i64>
}

// -----

// CHECK-LABEL: func @eval_divide
func.func @eval_divide() -> tensor<i64> {
  // CHECK-NOT: stablehlo.divide
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<1> : tensor<i64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.divide %0, %1 : tensor<i64>
  func.return %2 : tensor<i64>
}

// -----

// CHECK-LABEL: func @eval_get_dimension_size
func.func @eval_get_dimension_size(%arg0: tensor<4xf32>) -> tensor<i32> {
  // CHECK-NOT: stablehlo.eval_get_dimension_size
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<4xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @eval_maximum
func.func @eval_maximum() -> tensor<i64> {
  // CHECK-NOT: stablehlo.maximum
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<4> : tensor<i64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<4> : tensor<i64>
  %2 = stablehlo.maximum %0, %1 : tensor<i64>
  func.return %2 : tensor<i64>
}

// -----

// CHECK-LABEL: func @eval_multiply
func.func @eval_multiply() -> tensor<i64> {
  // CHECK-NOT: stablehlo.multiply
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<4> : tensor<i64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.multiply %0, %1 : tensor<i64>
  func.return %2 : tensor<i64>
}

// -----

// CHECK-LABEL: func @eval_remainder
func.func @eval_remainder() -> tensor<i64> {
  // CHECK-NOT: stablehlo.remainder
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.remainder %0, %1 : tensor<i64>
  func.return %2 : tensor<i64>
}

// -----

// CHECK-LABEL: func @eval_reshape
func.func @eval_reshape() -> tensor<1xi64> {
  // CHECK-NOT: stablehlo.reshape
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<1> : tensor<1xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.reshape %0 : (tensor<i64>) -> tensor<1xi64>
  func.return %1 : tensor<1xi64>
}

// -----

// CHECK-LABEL: func @eval_select
func.func @eval_select() -> tensor<2xi64> {
  // CHECK-NOT: stablehlo.select
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<[false, true]> : tensor<2xi1>
  %1 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %3 = stablehlo.select %0, %1, %2 : tensor<2xi1>, tensor<2xi64>
  func.return %3 : tensor<2xi64>
}

// -----

// CHECK-LABEL: func @eval_sign
func.func @eval_sign() -> tensor<3xi64> {
  // CHECK-NOT: stablehlo.sign
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<[-1, 0, 1]> : tensor<3xi64>
  %1 = stablehlo.sign %0 : (tensor<3xi64>) -> tensor<3xi64>
  func.return %1 : tensor<3xi64>
}

// -----

// CHECK-LABEL: func @eval_slice
func.func @eval_slice() -> tensor<2xi64> {
  // CHECK-NOT: stablehlo.slice
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %1 = "stablehlo.slice"(%0) {
    start_indices = dense<0> : tensor<1xi64>,
    limit_indices = dense<2> : tensor<1xi64>,
    strides = dense<1> : tensor<1xi64>
  } : (tensor<4xi64>) -> tensor<2xi64>
  func.return %1 : tensor<2xi64>
}

// -----

// CHECK-LABEL: func @eval_subtract
func.func @eval_subtract() -> tensor<i64> {
  // CHECK-NOT: stablehlo.subtract
  // CHECK: [[RESULT:%.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: return [[RESULT]]
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.subtract %0, %1 : tensor<i64>
  func.return %2 : tensor<i64>
}

// -----

// CHECK-LABEL: func @refine_all_gather_cross_replica
func.func @refine_all_gather_cross_replica(%arg0: tensor<4x4xf32>) -> tensor<4x?xf32> {
  // CHECK: "stablehlo.all_gather"{{.*}} -> tensor<4x16xf32>
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x4xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: func @refine_all_gather_cross_replica_and_partition
func.func @refine_all_gather_cross_replica_and_partition(%arg0: tensor<4x4xf32>) -> tensor<4x?xf32> {
  // CHECK: "stablehlo.all_gather"{{.*}} -> tensor<4x?xf32>
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x4xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: func @refine_all_gather_flattened_ids
func.func @refine_all_gather_flattened_ids(%arg0: tensor<4x4xf32>) -> tensor<4x?xf32> {
  // CHECK: "stablehlo.all_gather"{{.*}} -> tensor<4x16xf32>
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    use_global_device_ids
  } : (tensor<4x4xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: func @refine_bitcast_convert_different_bitwidths
func.func @refine_bitcast_convert_different_bitwidths(%arg0 : tensor<4xf32>) -> tensor<*xi8> {
  // CHECK: stablehlo.bitcast_convert{{.*}} -> tensor<*xi8>
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<*xi8>
  func.return %0 : tensor<*xi8>
}

// -----

// CHECK-LABEL: func @refine_bitcast_convert_same_bitwidth
func.func @refine_bitcast_convert_same_bitwidth(%arg0 : tensor<4xf32>) -> tensor<*xi32> {
  // CHECK: stablehlo.bitcast_convert{{.*}} -> tensor<4xi32>
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xf32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// TODO(#1037): Switch to *xi32 once fixed.
// CHECK-LABEL: func @refine_convert
func.func @refine_convert(%arg0 : tensor<4xf32>) -> tensor<?xi32> {
  // CHECK: stablehlo.convert{{.*}} -> tensor<4xi32>
  %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: @refine_convolution
func.func @refine_convolution(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) -> tensor<*xf32> {
  // CHECK: stablehlo.convolution{{.*}} -> tensor<100x28x28x1xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f],
    window = {
      stride = [1, 1],
      pad = [[2, 2], [2, 2]],
      lhs_dilate = [1, 1],
      rhs_dilate = [1, 1]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_custom_call
func.func @refine_custom_call(%arg0: tensor<4xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK: stablehlo.custom_call{{.*}} -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%arg0, %0, %1) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return %2#0, %2#1 : tensor<*xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_dot_general
func.func @refine_dot_general(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<*xf32> {
  // CHECK: stablehlo.dot_general{{.*}} -> tensor<2x4x5xf32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_dynamic_broadcast_in_dim
func.func @refine_dynamic_broadcast_in_dim(%arg0: tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim{{.*}} -> tensor<3x4xf32>
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_dynamic_conv
func.func @refine_dynamic_conv(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) -> tensor<*xf32> {
  // CHECK: stablehlo.dynamic_conv{{.*}} -> tensor<100x28x28x1xf32>
  %0 = stablehlo.constant dense<[[2, 2], [2, 2]]> : tensor<2x2xi32>
  %1 = "stablehlo.dynamic_conv"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    window_strides = dense<[1, 1]> : tensor<2xi64>,
    lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_dynamic_iota
func.func @refine_dynamic_iota() -> tensor<*xf32> {
  // CHECK: stablehlo.dynamic_iota{{.*}} -> tensor<4xf32>
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_dynamic_pad
func.func @refine_dynamic_pad(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  // CHECK: stablehlo.dynamic_pad{{.*}} -> tensor<6xf32>
  %0 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %1 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %2 = stablehlo.constant dense<[0]> : tensor<1xi64>
  %3 = stablehlo.dynamic_pad %arg0, %arg1, %0, %1, %2
           : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_dynamic_reshape
func.func @refine_dynamic_reshape(%arg0: tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: stablehlo.dynamic_reshape{{.*}} -> tensor<1x4xf32>
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_infer_type_op_interface_supported_dialect_chlo
func.func @refine_infer_type_op_interface_supported_dialect_chlo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: chlo.broadcast_add{{.*}} -> tensor<4xf32>
  %1 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// TODO(#1037): Switch to *xf32 once fixed.
// CHECK-LABEL: @refine_infer_type_op_interface_supported_dialect_stablehlo
func.func @refine_infer_type_op_interface_supported_dialect_stablehlo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.add{{.*}} : tensor<4xf32>
  %1 = stablehlo.add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @refine_real_dynamic_slice_using_dynamic_slice_non_unit_strides
func.func @refine_real_dynamic_slice_using_dynamic_slice_non_unit_strides(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<*xf32> {
  // CHECK: stablehlo.real_dynamic_slice{{.*}} -> tensor<*xf32>
  %0 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.constant dense<[2]> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %2
           : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_real_dynamic_slice_using_dynamic_slice_unit_strides
func.func @refine_real_dynamic_slice_using_dynamic_slice_unit_strides(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<*xf32> {
  // CHECK: stablehlo.real_dynamic_slice{{.*}} -> tensor<1xf32>
  %0 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %2
           : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_real_dynamic_slice_using_slice
func.func @refine_real_dynamic_slice_using_slice(%arg0: tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: stablehlo.real_dynamic_slice{{.*}} -> tensor<1xf32>
  %0 = stablehlo.constant dense<[0]> : tensor<1xi64>
  %1 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %2 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %0, %1, %2
           : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @refine_reduce_scatter_cross_replica
func.func @refine_reduce_scatter_cross_replica(%data: tensor<4x16xf32>) -> tensor<4x?xf32> {
  // CHECK: "stablehlo.reduce_scatter"{{.*}}
  // CHECK: -> tensor<4x4xf32>
  %0 = "stablehlo.reduce_scatter"(%data) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    scatter_dimension = 1 : i64
  } : (tensor<4x16xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @refine_reduce_scatter_cross_replica_and_partition
func.func @refine_reduce_scatter_cross_replica_and_partition(%data: tensor<4x16xf32>) -> tensor<4x?xf32> {
  // CHECK: "stablehlo.reduce_scatter"{{.*}}
  // CHECK: -> tensor<4x?xf32>
  %0 = "stablehlo.reduce_scatter"(%data) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    scatter_dimension = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @refine_reduce_scatter_flattened_ids
func.func @refine_reduce_scatter_flattened_ids(%data: tensor<4x16xf32>) -> tensor<4x?xf32> {
  // CHECK: "stablehlo.reduce_scatter"{{.*}}
  // CHECK: -> tensor<4x4xf32>
  %0 = "stablehlo.reduce_scatter"(%data) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    scatter_dimension = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    use_global_device_ids
  } : (tensor<4x16xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @refine_rng
func.func @refine_rng(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  // CHECK: stablehlo.rng{{.*}} -> tensor<4xf32>
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.rng %arg0, %arg1, %0, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// TODO(#1037): Switch to *x!quant.uniform<i8:f32, 1.000000e+00> once fixed.
// CHECK-LABEL: @refine_uniform_quantize
func.func @refine_uniform_quantize(%arg0 : tensor<4xf32>) -> tensor<?x!quant.uniform<i8:f32, 1.0>> {
  // CHECK: stablehlo.uniform_quantize{{.*}} -> tensor<4x!quant.uniform<i8:f32, 1.000000e+00>>
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<4xf32>) -> tensor<?x!quant.uniform<i8:f32, 1.0>>
  func.return %0 : tensor<?x!quant.uniform<i8:f32, 1.0>>
}

// -----

// CHECK-LABEL: @refine_while
func.func @refine_while(%arg0: tensor<4xf32>) -> tensor<*xf32> {
  // TODO(#871): Also check cond and body when fixed.
  // CHECK: stablehlo.while{{.*}} : tensor<4xf32>
  // CHECK: stablehlo.abs{{.*}} : tensor<4xf32>
  %0 = "stablehlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<?xf32>):
    %1 = stablehlo.constant dense<true> : tensor<i1>
    stablehlo.return %1 : tensor<i1>
  },  {
  ^bb0(%arg1: tensor<?xf32>):
    stablehlo.return %arg1 : tensor<?xf32>
  }) : (tensor<4xf32>) -> tensor<*xf32>
  %1 = stablehlo.abs %0 : tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// TODO: Implement support for these ops.
// * custom_call (#851).
// * dynamic_conv (#867).

// -----

// CHECK-LABEL: func @update_function_type
// CHECK-SAME: (%arg0: tensor<4xf32>) -> tensor<4xf32>
func.func @update_function_type(%arg0: tensor<4xf32>) -> tensor<*xf32> {
  // CHECK-NOT: tensor.cast
  %0 = tensor.cast %arg0 : tensor<4xf32> to tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @update_region_type
func.func @update_region_type(%arg0: tensor<i32>, %arg1: tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: "stablehlo.case"
  // CHECK: -> tensor<4xf32>
  // CHECK: stablehlo.abs{{.*}} : tensor<4xf32>
  %0 = "stablehlo.case"(%arg0) ({
    "stablehlo.return"(%arg1) : (tensor<4xf32>) -> ()
  }, {
    "stablehlo.return"(%arg1) : (tensor<4xf32>) -> ()
  }) : (tensor<i32>) -> tensor<*xf32>
  %1 = stablehlo.abs %0 : tensor<*xf32>
  return %1 : tensor<*xf32>
}
