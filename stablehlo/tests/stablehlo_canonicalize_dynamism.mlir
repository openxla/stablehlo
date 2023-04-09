// RUN: stablehlo-opt --stablehlo-canonicalize-dynamism --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @custom_call_success
func.func @custom_call_success(%arg0: tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%arg0, %0, %1) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  return %2#0, %2#1 : tensor<1x2xf32>, tensor<3x4xf32>
}

// -----

func.func @custom_call_failure_attr_number_of_elements(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+3{{indices_of_shape_operands: number of elements (2) must be equal to the number of operation results (1)}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2 = stablehlo.custom_call @foo(%arg0, %0, %1) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  return %2 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_attr_rank(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{indices_of_shape_operands: must have rank = 1}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[[1]]> : tensor<1x1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_attr_element_type(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{indices_of_shape_operands: must have i64 element type}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi32>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_out_of_bounds_operand_index(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{indices_of_shape_operands: index #0 (2) must be within bounds for operation operands (from 0 to 2)}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[2]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_incompatible_result_type(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{refinement #0 ([1, 1]) must be compatible with operation result ('tensor<1x2xf32>')}}
  %0 = stablehlo.constant dense<[1, 1]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_dynamic_shape_operand(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %arg1)
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}

// -----

func.func @custom_call_failure_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %0)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %1 : tensor<1x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_conv
func.func @dynamic_conv(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  //  CHECK-NOT: stablehlo.dynamic_conv
  //      CHECK: stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME:  dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f],
  // CHECK-SAME:  window = {
  // CHECK-SAME:    stride = [1, 1],
  // CHECK-SAME:    pad = {{\[}}[2, 2], [2, 2]],
  // CHECK-SAME:    lhs_dilate = [1, 1],
  // CHECK-SAME:    rhs_dilate = [1, 1]
  // CHECK-SAME: } {
  // CHECK-SAME:   batch_group_count = 1 : i64,
  // CHECK-SAME:   feature_group_count = 1 : i64
  // CHECK-SAME: } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32>
  %0 = stablehlo.constant dense<2> : tensor<2x2xi32>
  %1 = "stablehlo.dynamic_conv"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    window_strides = dense<1> : tensor<2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<100x28x28x1xf32>
  return %1 : tensor<100x28x28x1xf32>
}

// -----

// CHECK-LABEL: func @dynamic_iota
func.func @dynamic_iota() -> tensor<4xf32> {
  // CHECK-NOT: stablehlo.dynamic_iota
  // CHECK: stablehlo.iota dim = 0 : tensor<4xf32>
  %0 = stablehlo.constant dense<4> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim
func.func @dynamic_broadcast_in_dim(%arg0: tensor<4xf32>) -> tensor<3x4xf32> {
  // CHECK-NOT: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<3x4xf32>
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @dynamic_gather
func.func @dynamic_gather(%arg0 : tensor<2x4x9xi32>, %arg1 : tensor<1x5x2xi32>, %arg2 : tensor<3xi32>) -> tensor<1x5x8xi32> {
  //  CHECK-NOT: stablehlo.dynamic_gather
  //      CHECK: "stablehlo.gather"(%arg0, %arg1) {
  // CHECK-SAME:   dimension_numbers = #stablehlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  // CHECK-SAME: } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  %0 = stablehlo.constant dense<[1, 1, 8]> : tensor<3xi32>
  %1 = "stablehlo.dynamic_gather"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  return %1 : tensor<1x5x8xi32>
}

// -----

// CHECK-LABEL: func @dynamic_pad
func.func @dynamic_pad(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<6xf32> {
  // CHECK-NOT: stablehlo.dynamic_pad
  // CHECK: stablehlo.pad %arg0, %arg1, low = [1], high = [1], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<6xf32>
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.constant dense<0> : tensor<1xi64>
  %2 = stablehlo.dynamic_pad %arg0, %arg1, %0, %0, %1 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<6xf32>
  return %2 : tensor<6xf32>
}

// -----

// CHECK-LABEL: func @dynamic_reshape
func.func @dynamic_reshape(%arg0: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NOT: stablehlo.dynamic_reshape
  // CHECK: stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<1x4xf32>
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice
func.func @real_dynamic_slice_to_dynamic_slice(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: [[SIZE0_1D:%.*]] = "stablehlo.slice"(%arg1) {
  // CHECK-SAME:   limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-SAME:   start_indices = dense<0> : tensor<1xi64>,
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[SIZE0_0D:%.*]] = stablehlo.reshape [[SIZE0_1D]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: stablehlo.dynamic_slice %arg0, [[SIZE0_0D]], sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %0 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice
func.func @real_dynamic_slice_to_slice(%arg0: tensor<4xf32>) -> tensor<1xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: "stablehlo.slice"(%arg0) {
  // CHECK-SAME:   limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-SAME:   start_indices = dense<0> : tensor<1xi64>,
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<4xf32>) -> tensor<1xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %0, %1, %1 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}
