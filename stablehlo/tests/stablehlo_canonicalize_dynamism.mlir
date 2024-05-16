// RUN: stablehlo-opt --stablehlo-canonicalize-dynamism --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @custom_call_success_one_output
func.func @custom_call_success_one_output(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> tensor<1x2xf32>
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_multiple_outputs
func.func @custom_call_success_multiple_outputs(%arg0: tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%arg0, %0, %1) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  return %2#0, %2#1 : tensor<1x2xf32>, tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_mixed_positions
func.func @custom_call_success_mixed_positions(%arg0: tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%0, %arg0, %1) {
    indices_of_shape_operands = dense<[0, 2]> : tensor<2xi64>
  } : (tensor<2xi64>, tensor<4xf32>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  return %2#0, %2#1 : tensor<1x2xf32>, tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_mixed_positions_layouts
func.func @custom_call_success_mixed_positions_layouts(%arg0: tensor<4x3xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
  //      CHECK: stablehlo.custom_call @foo(%arg0) {
  // CHECK-SAME:   operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
  // CHECK-SAME:   result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]
  // CHECK-SAME: } : (tensor<4x3xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%0, %arg0, %1) {
    indices_of_shape_operands = dense<[0, 2]> : tensor<2xi64>,
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>],
    result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]
  } : (tensor<2xi64>, tensor<4x3xf32>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  return %2#0, %2#1 : tensor<1x2xf32>, tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_repeating_operands
func.func @custom_call_success_repeating_operands(%arg0: tensor<4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1:2 = stablehlo.custom_call @foo(%arg0, %0, %0) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  return %1#0, %1#1 : tensor<1x2xf32>, tensor<1x2xf32>
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
  // expected-error@+2{{refinement #0 ([1, 1]) must be compatible with operation result #0 ('tensor<1x2xf32>')}}
  %0 = stablehlo.constant dense<[1, 1]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call_inapplicable_dynamic_shape_operand
func.func @custom_call_inapplicable_dynamic_shape_operand(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %arg1)
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}

// -----

// CHECK-LABEL: func @custom_call_inapplicable_missing_indices_of_shape_operands
func.func @custom_call_inapplicable_missing_indices_of_shape_operands(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %c)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call_inapplicable_dynamic_result_type
func.func @custom_call_inapplicable_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %c)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %1 : tensor<1x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_success
func.func @dynamic_broadcast_in_dim_success(%arg0: tensor<4xf32>) -> tensor<3x4xf32> {
  // CHECK-NOT: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<3x4xf32>
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_inapplicable_dynamic_operand_type
func.func @dynamic_broadcast_in_dim_inapplicable_dynamic_operand_type(%arg0: tensor<?xf32>) -> tensor<3x4xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_inapplicable_dynamic_output_dimensions
func.func @dynamic_broadcast_in_dim_inapplicable_dynamic_output_dimensions(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<3x4xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_inapplicable_dynamic_result_type
func.func @dynamic_broadcast_in_dim_inapplicable_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<3x?xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x?xf32>
  return %1 : tensor<3x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_conv_success_static_result_type
func.func @dynamic_conv_success_static_result_type(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
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
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<100x28x28x1xf32>
  return %1 : tensor<100x28x28x1xf32>
}

// -----

// CHECK-LABEL: func @dynamic_conv_success_dynamic_result_type
func.func @dynamic_conv_success_dynamic_result_type(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>) -> tensor<?x28x28x1xf32> {
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
  // CHECK-SAME: } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<?x28x28x1xf32>
  %0 = stablehlo.constant dense<2> : tensor<2x2xi32>
  %1 = "stablehlo.dynamic_conv"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<?x28x28x1xf32>
  return %1 : tensor<?x28x28x1xf32>
}

// -----

// CHECK-LABEL: func @dynamic_conv_inapplicable_dynamic_padding
func.func @dynamic_conv_inapplicable_dynamic_padding(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<2x2xi32>) -> tensor<100x28x28x1xf32> {
  // CHECK: stablehlo.dynamic_conv
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<100x28x28x1xf32>
  return %0 : tensor<100x28x28x1xf32>
}

// -----

// CHECK-LABEL: @dynamic_gather_success_static_result_type
func.func @dynamic_gather_success_static_result_type(%arg0 : tensor<2x4x9xi32>, %arg1 : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  //  CHECK-NOT: stablehlo.dynamic_gather
  //      CHECK: "stablehlo.gather"(%arg0, %arg1) <{
  // CHECK-SAME:   dimension_numbers = #stablehlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   slice_sizes = array<i64: 1, 1, 8>
  // CHECK-SAME: }> : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
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

// CHECK-LABEL: @dynamic_gather_success_dynamic_result_type
func.func @dynamic_gather_success_dynamic_result_type(%arg0 : tensor<2x4x9xi32>, %arg1 : tensor<1x5x2xi32>) -> tensor<1x5x?xi32> {
  //  CHECK-NOT: stablehlo.dynamic_gather
  //      CHECK: "stablehlo.gather"(%arg0, %arg1) <{
  // CHECK-SAME:   dimension_numbers = #stablehlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   slice_sizes = array<i64: 1, 1, 8>
  // CHECK-SAME: }> : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x?xi32>
  %0 = stablehlo.constant dense<[1, 1, 8]> : tensor<3xi32>
  %1 = "stablehlo.dynamic_gather"(%arg0, %arg1, %0) <{
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >
  }> : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x?xi32>
  return %1 : tensor<1x5x?xi32>
}

// -----

// CHECK-LABEL: @dynamic_gather_inapplicable_dynamic_slice_sizes
func.func @dynamic_gather_inapplicable_dynamic_slice_sizes(%arg0 : tensor<2x4x9xi32>, %arg1 : tensor<1x5x2xi32>, %arg2 : tensor<3xi32>) -> tensor<1x5x8xi32> {
  // CHECK: stablehlo.dynamic_gather
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %arg2) <{
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >
  }> : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  return %0 : tensor<1x5x8xi32>
}

// -----

// CHECK-LABEL: func @dynamic_iota_success
func.func @dynamic_iota_success() -> tensor<4xf32> {
  // CHECK-NOT: stablehlo.dynamic_iota
  // CHECK: stablehlo.iota dim = 0 : tensor<4xf32>
  %0 = stablehlo.constant dense<4> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_iota_inapplicable_dynamic_output_shape
func.func @dynamic_iota_inapplicable_dynamic_output_shape(%arg0: tensor<1xi64>) -> tensor<4xf32> {
  // CHECK: stablehlo.dynamic_iota
  %0 = stablehlo.dynamic_iota %arg0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_iota_inapplicable_dynamic_result_type
func.func @dynamic_iota_inapplicable_dynamic_result_type() -> tensor<?xf32> {
  // CHECK: stablehlo.dynamic_iota
  %0 = stablehlo.constant dense<4> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_success_static_result_type
func.func @dynamic_pad_success_static_result_type(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<11xf32> {
  // CHECK-NOT: stablehlo.dynamic_pad
  // CHECK: stablehlo.pad %arg0, %arg1, low = [0], high = [1], interior = [2] : (tensor<4xf32>, tensor<f32>) -> tensor<11xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.dynamic_pad %arg0, %arg1, %0, %1, %2 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %3 : tensor<11xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_success_dynamic_result_type
func.func @dynamic_pad_success_dynamic_result_type(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  // CHECK-NOT: stablehlo.dynamic_pad
  // CHECK: stablehlo.pad %arg0, %arg1, low = [0], high = [1], interior = [2] : (tensor<4xf32>, tensor<f32>) -> tensor<?xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.dynamic_pad %arg0, %arg1, %0, %1, %2 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_inapplicable_dynamic_low
func.func @dynamic_pad_inapplicable_dynamic_low(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1xi64>) -> tensor<11xf32> {
  // CHECK: stablehlo.dynamic_pad
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.dynamic_pad %arg0, %arg1, %arg2, %0, %1 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %2 : tensor<11xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_inapplicable_dynamic_high
func.func @dynamic_pad_inapplicable_dynamic_high(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1xi64>) -> tensor<11xf32> {
  // CHECK: stablehlo.dynamic_pad
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.dynamic_pad %arg0, %arg1, %0, %arg2, %1 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %2 : tensor<11xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_inapplicable_dynamic_interior
func.func @dynamic_pad_inapplicable_dynamic_interior(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1xi64>) -> tensor<11xf32> {
  // CHECK: stablehlo.dynamic_pad
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.dynamic_pad %arg0, %arg1, %0, %1, %arg2 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %2 : tensor<11xf32>
}

// -----

// CHECK-LABEL: func @dynamic_reshape_success
func.func @dynamic_reshape_success(%arg0: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NOT: stablehlo.dynamic_reshape
  // CHECK: stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<1x4xf32>
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_reshape_inapplicable_dynamic_output_shape
func.func @dynamic_reshape_inapplicable_dynamic_output_shape(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<1x4xf32> {
  // CHECK: stablehlo.dynamic_reshape
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_reshape_inapplicable_dynamic_result_type
func.func @dynamic_reshape_inapplicable_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.dynamic_reshape
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %1 : tensor<1x?xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_success_static_result_type
func.func @real_dynamic_slice_to_dynamic_slice_success_static_result_type(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: [[SIZE0_1D:%.*]] = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[SIZE0_0D:%.*]] = stablehlo.reshape [[SIZE0_1D]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: stablehlo.dynamic_slice %arg0, [[SIZE0_0D]], sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.constant dense<1> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %3 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_success_dynamic_result_type
func.func @real_dynamic_slice_to_dynamic_slice_success_dynamic_result_type(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<?xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: [[SIZE0_1D:%.*]] = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[SIZE0_0D:%.*]] = stablehlo.reshape [[SIZE0_1D]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: stablehlo.dynamic_slice %arg0, [[SIZE0_0D]], sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<?xf32>
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.constant dense<1> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_inapplicable_non_unit_strides
func.func @real_dynamic_slice_to_dynamic_slice_inapplicable_non_unit_strides(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %arg2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_inapplicable_unsupported_limit
func.func @real_dynamic_slice_to_dynamic_slice_inapplicable_unsupported_limit(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>, %arg3: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.real_dynamic_slice %arg0, %arg1, %arg2, %arg3 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_success_static_result_type
func.func @real_dynamic_slice_to_slice_success_static_result_type(%arg0: tensor<4xf32>) -> tensor<1xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: stablehlo.slice %arg0 [0:1:2] : (tensor<4xf32>) -> tensor<1xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %0, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %3 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_success_dynamic_result_type
func.func @real_dynamic_slice_to_slice_success_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<?xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: stablehlo.slice %arg0 [0:1:2] : (tensor<4xf32>) -> tensor<?xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %0, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_inapplicable_dynamic_start
func.func @real_dynamic_slice_to_slice_inapplicable_dynamic_start(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %arg1, %0, %1 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_inapplicable_dynamic_limit
func.func @real_dynamic_slice_to_slice_inapplicable_dynamic_limit(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %0, %arg1, %1 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_inapplicable_dynamic_strides
func.func @real_dynamic_slice_to_slice_inapplicable_dynamic_strides(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %0, %1, %arg1 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}
