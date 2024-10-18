// RUN: stablehlo-opt --stablehlo-aggressive-simplification --allow-unregistered-dialect --split-input-file %s | FileCheck %s

/////////
// AddOp

// CHECK-LABEL: @add_cst_on_rhs
func.func @add_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  // CHECK: stablehlo.add %arg0, %cst : tensor<f32>
  %0 = stablehlo.add %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @add_zero_like_lhs
func.func @add_zero_like_lhs(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.add %0, %arg0 : tensor<i32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0
  return %1 : tensor<i32>
}

// CHECK-LABEL: @add_zero_like_rhs
func.func @add_zero_like_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.add %arg0, %0 : tensor<f32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0
  return %1 : tensor<f32>
}

// -----

/////////
// AndOp

// CHECK-LABEL: @and_cst_on_rhs
func.func @and_cst_on_rhs(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<true> : tensor<2xi1>
  %0 = stablehlo.and %cst, %arg0 : tensor<2xi1>
  // Check that constant canonicalized to RHS, then other patterns apply
  // CHECK-NOT: stablehlo.and
  // return %arg0
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: @and_zero
func.func @and_zero(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<false> : tensor<2xi1>
  %1 = stablehlo.and %0, %arg0 : tensor<2xi1>
  // CHECK-NOT: stablehlo.and
  // CHECK: [[FALSE:%.+]] = stablehlo.constant dense<false> : tensor<2xi1>
  // CHECK: return [[FALSE]]
  return %1 : tensor<2xi1>
}

// CHECK-LABEL: @and_one
func.func @and_one(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<true> : tensor<2xi1>
  %1 = stablehlo.and %0, %arg0 : tensor<2xi1>
  // CHECK-NOT: stablehlo.and
  // CHECK: return %arg0
  return %1 : tensor<2xi1>
}

// -----

/////////
// BroadcastInDim

// CHECK-LABEL: func.func @broadcast_in_dim_transpose
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x3xi32>)
func.func @broadcast_in_dim_transpose(%arg0: tensor<3x3xi32>)
  -> (tensor<3x3xi32>, tensor<3x3xi32>) {
  %3 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<3x3xi32>) -> tensor<3x3xi32>
  %4 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3xi32>

  // CHECK: [[R4:%.+]] = stablehlo.transpose [[ARG0]], dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3xi32>

  // CHECK-NEXT: return [[ARG0]], [[R4]]
  return %3, %4 : tensor<3x3xi32>, tensor<3x3xi32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_nested
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x3xi32>)
func.func @broadcast_in_dim_nested(%arg0: tensor<3x3xi32>)
  -> (tensor<3x2x3x3xi32>) {
  %6 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<3x3xi32>) -> tensor<3x3x2xi32>
  %7 = stablehlo.broadcast_in_dim %6, dims = [0, 2, 1] : (tensor<3x3x2xi32>) -> tensor<3x2x3x3xi32>
  // CHECK: [[R6:%.+]] = stablehlo.broadcast_in_dim [[ARG0]], dims = [2, 0] : (tensor<3x3xi32>) -> tensor<3x2x3x3xi32>

  // CHECK-NEXT: return [[R6]]
  return %7 : tensor<3x2x3x3xi32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_reshape
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<3x6xi32>)
func.func @broadcast_in_dim_reshape(%arg0: tensor<3x6xi32>)
  -> (tensor<1x3x6xi32>, tensor<3x6x1xi32>) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<3x6xi32>) -> tensor<1x3x6xi32>
  %5 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<3x6xi32>) -> tensor<3x6x1xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<3x6xi32>) -> tensor<1x3x6xi32>
  // CHECK-DAG:  [[R5:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<3x6xi32>) -> tensor<3x6x1xi32>

  // CHECK-NEXT: return [[R0]], [[R5]]
  return %0, %5 : tensor<1x3x6xi32>, tensor<3x6x1xi32>
}

// CHECK-LABEL: func.func @broadcast_in_dim_not_identity_broadcasts
func.func @broadcast_in_dim_not_identity_broadcasts(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.broadcast_in_dim
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

/////////
// CompareOp

// CHECK-LABEL: func.func @compare_signed_arg
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>)
func.func @compare_signed_arg(%arg0: tensor<i32>)
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare LE, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare NE, %arg0, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare NE, %c5, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-DAG:  [[C5:%.+]] = stablehlo.constant dense<5> : tensor<i32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.compare GT, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.compare LE, [[ARG0]], [[C5]], SIGNED
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.compare NE, [[ARG0]], [[C5]], SIGNED

  // CHECK-NEXT: return [[C1]], [[C0]], [[C1]], [[C0]], [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-LABEL: func.func @compare_unsigned_arg
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>)
func.func @compare_unsigned_arg(%arg0: tensor<i32>)
  -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %c4 = stablehlo.constant dense<4> : tensor<i32>
  %c5 = stablehlo.constant dense<5> : tensor<i32>

  %0 = stablehlo.compare EQ, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare LE, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare NE, %arg0, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %4 = stablehlo.compare EQ, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare GE, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare NE, %c5, %arg0, UNSIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-DAG:  [[C5:%.+]] = stablehlo.constant dense<5> : tensor<i32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.compare GT, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.compare LE, [[ARG0]], [[C5]], UNSIGNED
  // CHECK-DAG:  [[R3:%.+]] = stablehlo.compare NE, [[ARG0]], [[C5]], UNSIGNED

  // CHECK-NEXT: return [[C1]], [[C0]], [[C1]], [[C0]], [[R0]], [[R1]], [[R2]], [[R3]]
  return %0, %1, %2, %3, %4, %5, %6, %7 :
         tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}

// -----

/////////
// ConvertOp

// CHECK-LABEL: func.func @convert
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>)
func.func @convert(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %r = stablehlo.convert %arg0 : tensor<2xf32>

  // CHECK: return [[ARG0]]
  return %r : tensor<2xf32>
}

// -----

/////////
// DynamicBroadcastInDimOp

// CHECK-LABEL: func @dynamic_broadcast_in_dim_all_dims_non_expanding
func.func @dynamic_broadcast_in_dim_all_dims_non_expanding(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  // CHECK-NEXT: return %[[ARG]]
  %1 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = array<i64: 0>,
    known_expanding_dimensions = array<i64>,
    known_nonexpanding_dimensions = array<i64: 0>
  } : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// CHECK-LABEL: @dynamic_broadcast_of_dynamic_reshape_same_shape
func.func @dynamic_broadcast_of_dynamic_reshape_same_shape(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %0, %arg1, dims = [0, 1] : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>

  // CHECK-NOT: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.dynamic_reshape %arg0, %arg1
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<5x4xf32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape(%arg0: tensor<i32>) -> tensor<4x32xi32> {
  %0 = stablehlo.constant dense<[4, 32]> : tensor<2xi32>
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<i32>) -> tensor<4x32xi32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<i32>, tensor<2xi32>) -> tensor<?x32xi32>
  %2 = stablehlo.dynamic_reshape %1, %0 : (tensor<?x32xi32>, tensor<2xi32>) -> tensor<4x32xi32>
  // CHECK: return %[[RESULT]] : tensor<4x32xi32>
  func.return %2 : tensor<4x32xi32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape(%arg0: tensor<f32>) -> tensor<4x32xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[RESULT:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x32xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
  %2 = stablehlo.dynamic_reshape %1, %0 : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<4x32xf32>
  // CHECK: return %[[RESULT]] : tensor<4x32xf32>
  func.return %2 : tensor<4x32xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast(%arg0: tensor<f32>) -> tensor<?x?xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<4x32xf32>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.*]] = stablehlo.convert %[[BCAST]] : (tensor<4x32xf32>) -> tensor<?x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// -----

/////////
// DynamicReshapeOp

// CHECK-LABEL: func.func @dynamic_reshape
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1xf32>, [[ARG1:%.+]]: tensor<?x?xf32>, [[ARG2:%.+]]: tensor<2xi32>)
func.func @dynamic_reshape(%arg0: tensor<1xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<2xi32>)
          -> (tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x2xi32>) {
  %c0 = stablehlo.constant dense<[2, 1]> : tensor<2xi32>

  %0 = stablehlo.dynamic_reshape %arg0, %arg2 : (tensor<1xf32>, tensor<2xi32>) -> tensor<1x1xf32>
  %1 = stablehlo.dynamic_reshape %arg1, %c0 : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  %2 = stablehlo.dynamic_reshape %arg2, %arg2 : (tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.reshape [[ARG0]] : (tensor<1xf32>) -> tensor<1x1xf32>
  // CHECK-DAG:  [[R1:%.+]] = stablehlo.reshape [[ARG1]] : (tensor<?x?xf32>) -> tensor<2x1xf32>
  // CHECK-DAG:  [[R2:%.+]] = stablehlo.reshape [[ARG2]] : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK-NEXT: return [[R0]], [[R1]], [[R2]]
  return %0, %1, %2 : tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x2xi32>
}

// -----

/////////
// GatherOp

// CHECK-LABEL: func.func @gather_to_slice
func.func @gather_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<3x6x5xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 3, 6, 5>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6x5xf32>
  return %1 : tensor<3x6x5xf32>
  // CHECK:      %[[RET:.*]] = stablehlo.slice %arg0 [1:4, 0:6, 2:7]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<3x6x5xf32>
  // CHECK-NEXT: return %[[RET]] : tensor<3x6x5xf32>
}

// CHECK-LABEL: func.func @gather_scalar_index_to_slice
func.func @gather_scalar_index_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x4xf32> {
  %0 = arith.constant dense<1> : tensor<i32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 5, 6, 4>} : (tensor<5x6x7xf32>, tensor<i32>) -> tensor<5x6x4xf32>
  return %1 : tensor<5x6x4xf32>
  // CHECK:      %[[RET:.*]] = stablehlo.slice %arg0 [0:5, 0:6, 1:5]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<5x6x4xf32>
  // CHECK-NEXT: return %[[RET]] : tensor<5x6x4xf32>
}

// CHECK-LABEL: func.func @gather_to_slice_reshape
func.func @gather_to_slice_reshape(%arg0: tensor<5x6x7xf32>) -> tensor<3x6xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [2],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 3, 6, 1>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [1:4, 0:6, 2:3]
  // CHECK-SAME:    : (tensor<5x6x7xf32>) -> tensor<3x6x1xf32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<3x6x1xf32>) -> tensor<3x6xf32>
  // CHECK-NEXT: return %[[V1]] : tensor<3x6xf32>
}

// CHECK-LABEL: func.func @gather_to_slice_indices_clamp_upperbound
func.func @gather_to_slice_indices_clamp_upperbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<4> : tensor<1xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = array<i64: 1, 2>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  return %1 : tensor<2xui32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [3:4, 0:2]
  // CHECK-SAME:    : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK-NEXT: return %[[V1]] : tensor<2xui32>
}

// CHECK-LABEL: func.func @gather_to_slice_indices_clamp_lowerbound
func.func @gather_to_slice_indices_clamp_lowerbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<-1> : tensor<1xi32>
  %1 = "stablehlo.gather"(%arg0, %0) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = array<i64: 1, 2>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  return %1 : tensor<2xui32>
  // CHECK:      %[[V0:.*]] = stablehlo.slice %arg0 [0:1, 0:2]
  // CHECK-SAME:    : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK-NEXT: %[[V1:.*]] = stablehlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK-NEXT: return %[[V1]] : tensor<2xui32>
}

// -----

/////////
// GetDimensionSizeOp

// CHECK-LABEL: func.func @get_dimension_size
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x2x3xf32>, [[ARG1:%.+]]: tensor<?x2xf32>)
func.func @get_dimension_size(%arg0: tensor<1x2x3xf32>, %arg1: tensor<?x2xf32>)
          -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %a = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<1x2x3xf32>) -> tensor<i32>
  %b = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x2x3xf32>) -> tensor<i32>
  %c = stablehlo.get_dimension_size %arg0, dim = 2 : (tensor<1x2x3xf32>) -> tensor<i32>

  %d = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  %e = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<?x2xf32>) -> tensor<i32>

  // CHECK-DAG:  [[CST1:%.+]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-DAG:  [[CST2:%.+]] = stablehlo.constant dense<2> : tensor<i32>
  // CHECK-DAG:  [[CST3:%.+]] = stablehlo.constant dense<3> : tensor<i32>
  // CHECK-DAG:  [[DYN:%.+]]  = stablehlo.get_dimension_size [[ARG1]], dim = 0 : (tensor<?x2xf32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST1]], [[CST2]], [[CST3]], [[DYN]], [[CST2]]
  return %a, %b, %c, %d, %e : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

/////////
// GetTupleElementOp

// CHECK-LABEL: func.func @get_tuple_element
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tuple<tensor<f32>, tensor<f16>>)
func.func @get_tuple_element(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tuple<tensor<f32>, tensor<f16>>)
          -> (tensor<f32>, tensor<i32>, tensor<f16>) {
  %t = stablehlo.tuple %arg0, %arg1 : tuple<tensor<f32>, tensor<i32>>

  %a = stablehlo.get_tuple_element %t[0] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  %b = stablehlo.get_tuple_element %t[1] : (tuple<tensor<f32>, tensor<i32>>) -> tensor<i32>

  %c = stablehlo.get_tuple_element %arg2[1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>

  // CHECK:      [[GTE:%.+]] = stablehlo.get_tuple_element [[ARG2]][1] : (tuple<tensor<f32>, tensor<f16>>) -> tensor<f16>
  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[GTE]]
  return %a, %b, %c : tensor<f32>, tensor<i32>, tensor<f16>
}

// -----

/////////
// ImagOp / RealOp

// CHECK-LABEL: func.func @complex
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xf32>, [[ARG1:%.+]]: tensor<2xf32>)
func.func @complex(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %c = stablehlo.complex %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  %r = stablehlo.real %c : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %i = stablehlo.imag %c : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[ARG0]], [[ARG1]]
  return %r, %i : tensor<2xf32>, tensor<2xf32>
}

// -----

/////////
// MaxOp

// CHECK-LABEL: @maximum_cst_on_rhs
func.func @maximum_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.maximum %arg0, %cst : tensor<f32>
  %0 = stablehlo.maximum %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

/////////
// MinOp

// CHECK-LABEL: @minimum_cst_on_rhs
func.func @minimum_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.minimum %arg0, %cst : tensor<f32>
  %0 = stablehlo.minimum %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

/////////
// MulOp

// CHECK-LABEL: @multiply_cst_on_rhs
func.func @multiply_cst_on_rhs(%arg0: tensor<f32>) -> tensor<f32> {
  %cst = stablehlo.constant dense<2.0> : tensor<f32>
  // CHECK: stablehlo.multiply %arg0, %cst : tensor<f32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @multiply_by_zero
func.func @multiply_by_zero(%arg0: tensor<i32>) -> tensor<i32> {
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @multiply_by_one
func.func @multiply_by_one(%arg0: tensor<i32>) -> tensor<i32> {
  %cst = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.multiply %cst, %arg0 : tensor<i32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

/////////
// OrOp

// CHECK-LABEL: @or_cst_on_rhs
func.func @or_cst_on_rhs(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<false> : tensor<2xi1>
  %0 = stablehlo.or %cst, %arg0 : tensor<2xi1>
  // Check that constant canonicalized to RHS, then other patterns apply
  // CHECK-NOT: stablehlo.or
  // CHECK: return %arg0
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: @or_zero
func.func @or_zero(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<false> : tensor<2xi1>
  %1 = stablehlo.or %0, %arg0 : tensor<2xi1>

  // CHECK-NOT: stablehlo.or
  // CHECK: return %arg0
  return %1 : tensor<2xi1>
}

// CHECK-LABEL: @or_one
func.func @or_one(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = stablehlo.constant dense<true> : tensor<2xi1>
  %1 = stablehlo.or %0, %arg0 : tensor<2xi1>

  // CHECK-NOT: stablehlo.or
  // CHECK: [[TRUE:%.+]] = stablehlo.constant dense<true> : tensor<2xi1>
  // CHECK: return [[TRUE]]
  return %1 : tensor<2xi1>
}

// -----

/////////
// ReduceOp

// CHECK-LABEL: @reduce_no_dimensions
func.func @reduce_no_dimensions(%arg0: tensor<8xi64>, %arg1: tensor<8xi64>) -> (tensor<8xi64>, tensor<8xi64>) {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %0:2 = stablehlo.reduce(%arg0 init: %c), (%arg1 init: %c) across dimensions = [] : (tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>) -> (tensor<8xi64>, tensor<8xi64>)
    reducer(%arg2: tensor<i64>, %arg4: tensor<i64>) (%arg3: tensor<i64>, %arg5: tensor<i64>)  {
    %1 = stablehlo.add %arg2, %arg4 : tensor<i64>
    %2 = stablehlo.subtract %arg3, %arg5 : tensor<i64>
    stablehlo.return %1, %2 : tensor<i64>, tensor<i64>
  }
  // CHECK-NOT: stablehlo.reduce
  // CHECK: return %arg0, %arg1
  return %0#0, %0#1 : tensor<8xi64>, tensor<8xi64>
}

// CHECK-LABEL: func.func @reduce_noop_2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<4x8xi32>, [[ARG1:%.+]]: tensor<i32>)
func.func @reduce_noop_2(%arg0: tensor<4x8xi32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.reduce(%arg0 init: %0) across dimensions = [0, 1] : (tensor<4x8xi32>, tensor<i32>) -> tensor<i32>
    reducer(%b1: tensor<i32>, %b2: tensor<i32>) {
    stablehlo.return %arg1 : tensor<i32>
  }
  // CHECK: return [[ARG1]] : tensor<i32>
  func.return %1 : tensor<i32>
}


// Each reduce_unused_* test case is accompanied by an ASCII diagram that
// represents the surveyed reduce operation in a compact form:
//
//  (a, a1) (b, b1)
//    ▴       ▴
//    │       │
//    │       │
//    │       │
//    │       │
//    r0      r1
//    U
//
// In this case:
// a,  b  - are the operands to be reduced, i.e. %arg0, %arg1.
// a1, b1 - are the initial values of the operands, i.e. %0, %1.
// r0, r1 - are the results of stablehlo.reduce and/or stablehlo.return operation
//          (they are equivalent in this context), i.e %2#0, %2#1 and %3, %4.
// Arrows show which results depend on which inputs. More specifically it
// represents a set of instructions followed along the def-use chain starting
// from the return operand and a set of reachable operand pairs that conclude
// these chains:
// r0 is an alias for %3, set of instructions { %3 }
//                        set of reachable operand pairs { (%arg2, %arg3) }
// r1 is an alias for %4, set of instructions { %4 }
//                        set of reachable operand pairs { (%arg4, %arg5) }
// U below the result means it is used (live).
//
// To drop r1 use pair (b, b1).

// CHECK-LABEL: func.func @reduce_unused_case0
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case0(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG0]] init: [[R0]]) applies stablehlo.add
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>)
   reducer(%arg2: tensor<i64>, %arg3: tensor<i64>)
          (%arg4: tensor<i64>, %arg5: tensor<i64>)
  {
    %3 = stablehlo.add %arg2, %arg3 : tensor<i64>
    %4 = stablehlo.subtract %arg4, %arg5 : tensor<i64>
    stablehlo.return %3, %4 : tensor<i64>, tensor<i64>
  }
  return %2#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    │       │       │
//    │       │       │
//    │       │       │
//    r0      r1      r2
//    U               U
//
// To drop r1 use (b, b1).

// CHECK-LABEL: func.func @reduce_unused_case1
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case1(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:2 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.maximum %arg5, %arg8 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#2 : tensor<i64>, tensor<i64>
}
// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴
//    │       │
//    │       │
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//    U       U       U
//
// All results are used.

// CHECK-LABEL: func.func @reduce_unused_case2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case2(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#1, %3#2 : tensor<i64>, tensor<i64>, tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴
//    │       │
//    │       │
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//                    U
//
// To drop r0 and r1 use pairs (a, a1) (c, c1).

// CHECK-LABEL: func.func @reduce_unused_case3
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case3(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG1]] init: [[R0]]) applies stablehlo.maximum
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#2 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    ├───────┼───────┘
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//    U               U
//
// There is no suitable pair to drop r1.

// CHECK-LABEL: func.func @reduce_unused_case4
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case4(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %4, %arg8 : tensor<i64>
    %6 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %7 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %5, %6, %7 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#2 : tensor<i64>, tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    ├───────┼───────┘
//    │       │◄──────┐
//    │       │       │
//    r0      r1      r2
//    U
//
// There is 1 suitable pair (b, b1), but used by 2 return operands.

// CHECK-LABEL: func.func @reduce_unused_case5
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case5(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %4, %arg8 : tensor<i64>
    %6 = stablehlo.minimum %arg4, %arg7 : tensor<i64>
    %7 = stablehlo.maximum %arg4, %arg7 : tensor<i64>
    stablehlo.return %5, %6, %7 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴               ▴
//    │               │
//    │       ┌───────┘
//    │◄──────┼───────┐
//    │       │       │
//    r0      r1      r2
//    U
//
// Both r1 and r2 can be dropped using (c, c1) & (b, b1) pairs.

// CHECK-LABEL: func.func @reduce_unused_case6
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case6(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG0]] init: [[R0]]) applies stablehlo.add
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.minimum %arg3, %arg6 : tensor<i64>
    %6 = stablehlo.maximum %arg5, %arg8 : tensor<i64>
    stablehlo.return %4, %6, %5 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    │       ├──────►│
//    │       │◄──────┤
//    │       │       │
//    r0      r1      r2
//    U
//
// Both r1 and r2 can be dropped.

// CHECK-LABEL: func.func @reduce_unused_case7
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case7(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.reduce([[ARG0]] init: [[R0]]) applies stablehlo.add
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.multiply %arg5, %arg8 : tensor<i64>
    %7 = stablehlo.minimum %5, %6 : tensor<i64>
    %8 = stablehlo.maximum %5, %6 : tensor<i64>
    stablehlo.return %4, %7, %8 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       ├──────►│
//    │       │◄──────│
//    │◄──────┼───────┤
//    │       │       │
//    r0      r1      r2
//            U
//
// Non-conservative case. In theory r0 can be removed once r2 is removed, since
// they share a common dependency (a, a1). However, we cannot set synthetic
// return value, e.g. 0, instead of r2.

// CHECK-LABEL: func.func @reduce_unused_case8
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case8(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.multiply %arg5, %arg8 : tensor<i64>
    %7 = stablehlo.minimum %5, %6 : tensor<i64>
    %8 = stablehlo.maximum %5, %6 : tensor<i64>
    %9 = stablehlo.subtract %8, %4 : tensor<i64>
    stablehlo.return %4, %7, %9 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#1 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴       ▴
//    │       │       │
//    ├───────┼──────►│
//    │       │       │
//    │       │       │
//    r0      r1      r2
//    U
//
// Non-conservative case: |{used results}| < |{used operand pairs}|.
//                                  |{r0}| < |{(a, a1), (c, c1)}|
// r1 can be dropped, but partial pruning is not implemented.

// CHECK-LABEL: func.func @reduce_unused_case9
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case9(%arg0: tensor<8xi64>,
                               %arg1: tensor<8xi64>,
                               %arg2: tensor<8xi64>) -> tensor<i64> {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg5 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    %6 = stablehlo.multiply %arg5, %arg8 : tensor<i64>
    stablehlo.return %4, %5, %6 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0 : tensor<i64>
}

// -----

//  (a, a1) (b, b1) (c, c1)
//    ▴       ▴
//    │       │
//    │◄──────┼───────┐
//    │       │       │
//    │       │       │
//    r0      r1      r2
//    U               U
//
// Non-conservative case: |{used results}| > |{used operand pairs}|.
//                              |{r0, r2}| > |{(a, a1)}|
// r1 can be dropped, but partial pruning is not implemented.

// CHECK-LABEL: func.func @reduce_unused_case10
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<8xi64>, [[ARG1:%.+]]: tensor<8xi64>, [[ARG2:%.+]]: tensor<8xi64>)
func.func @reduce_unused_case10(%arg0: tensor<8xi64>,
                                %arg1: tensor<8xi64>,
                                %arg2: tensor<8xi64>) -> (tensor<i64>, tensor<i64>) {
  // CHECK: [[R0:%.+]] = stablehlo.constant dense<1>
  // CHECK: [[R1:%.+]] = stablehlo.constant dense<2>
  // CHECK: [[R2:%.+]] = stablehlo.constant dense<3>
  // CHECK: [[R3:%.+]]:3 = stablehlo.reduce([[ARG0]] init: [[R0]]), ([[ARG1]] init: [[R1]]), ([[ARG2]] init: [[R2]])
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.constant dense<2> : tensor<i64>
  %2 = stablehlo.constant dense<3> : tensor<i64>
  %3:3 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1), (%arg2 init: %2) across dimensions = [0] :
  (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>, tensor<i64>, tensor<i64>, tensor<i64>) ->
  (tensor<i64>, tensor<i64>, tensor<i64>)
   reducer(%arg3: tensor<i64>, %arg6: tensor<i64>)
          (%arg4: tensor<i64>, %arg7: tensor<i64>)
          (%arg5: tensor<i64>, %arg8: tensor<i64>)
  {
    %4 = stablehlo.add %arg3, %arg6 : tensor<i64>
    %5 = stablehlo.subtract %arg4, %arg7 : tensor<i64>
    stablehlo.return %4, %5, %4 : tensor<i64>, tensor<i64>, tensor<i64>
  }
  return %3#0, %3#2 : tensor<i64>, tensor<i64>
}

// -----

/////////
// ReshapeOp

// CHECK-LABEL: func @reshape_identity
func.func @reshape_identity(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
  %0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NOT: stablehlo.reshape
  // CHECK: return %arg0
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @reshape_reshape
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
func.func @reshape_reshape(%arg0: tensor<4x4xi32>) -> tensor<16xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<4x4xi32>) -> tensor<2x8xi32>
  %1 = stablehlo.reshape %0 : (tensor<2x8xi32>) -> tensor<16xi32>
  // CHECK: [[R0:%.+]] = stablehlo.reshape %[[ARG0]] : (tensor<4x4xi32>) -> tensor<16xi32>
  return %1 : tensor<16xi32>
}

// -----

/////////
// SubtractOp

// CHECK-LABEL: @subtract_same_lhs_rhs
func.func @subtract_same_lhs_rhs(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: stablehlo.constant dense<0> : tensor<2xi32>
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: @subtract_zero
func.func @subtract_zero(%arg0: tensor<2xi32>, %arg1: tensor<2xf32>) -> (tensor<2xi32>, tensor<2xf32>) {
  %0 = stablehlo.constant dense<0> : tensor<2xi32>
  %1 = stablehlo.subtract %arg0, %0 : tensor<2xi32>
  %2 = stablehlo.constant dense<0.0> : tensor<2xf32>
  %3 = stablehlo.subtract %arg1, %2 : tensor<2xf32>
  // CHECK-NOT: stablehlo.constant
  // CHECK: return %arg0, %arg1
  return %1, %3: tensor<2xi32>, tensor<2xf32>
}

// -----

/////////
// SelectOp

// CHECK-LABEL: func.func @select
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>, [[ARGC:%.+]]: tensor<2xi1>)
func.func @select(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %argC: tensor<2xi1>)
  -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>) {
  %c0 = stablehlo.constant dense<false> : tensor<i1>
  %c1 = stablehlo.constant dense<true> : tensor<i1>

  %c0_2 = stablehlo.constant dense<false> : tensor<2xi1>
  %c1_2 = stablehlo.constant dense<true> : tensor<2xi1>

  %cond = stablehlo.constant dense<[false, true, false, true]> : tensor<4xi1>
  %foo = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %bar = stablehlo.constant dense<[5, 6, 7, 8]> : tensor<4xi32>

  %0 = stablehlo.select %argC, %arg0, %arg0 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = stablehlo.select %c0, %arg0, %arg1 : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %2 = stablehlo.select %c1, %arg0, %arg1 : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %3 = stablehlo.select %c0_2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %4 = stablehlo.select %c1_2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %5 = stablehlo.select %argC, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>

  %6 = stablehlo.select %cond, %foo, %bar : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-DAG:  [[R0:%.+]] = stablehlo.select [[ARGC]], [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[C0:%.+]] = stablehlo.constant dense<[5, 2, 7, 4]> : tensor<4xi32>

  // CHECK-NEXT: return [[ARG0]], [[ARG1]], [[ARG0]], [[ARG1]], [[ARG0]], [[R0]], [[C0]]
  return %0, %1, %2, %3, %4, %5, %6 :
         tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>
}

// CHECK-LABEL: func.func @select_into_minmax1
// CHECK-SAME:   [[ARG0:%.+]]: tensor<2xi32>, [[ARG1:%.+]]: tensor<2xi32>, [[ARG2:%.+]]: tensor<2xi32>, [[ARG3:%.+]]: tensor<2xi32>)
func.func @select_into_minmax1(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>,
                               %arg2: tensor<2xi32>, %arg3: tensor<2xi32>)
  -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {

  %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %1 = stablehlo.compare NE, %arg0, %arg1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %2 = stablehlo.compare GE, %arg0, %arg1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %3 = stablehlo.compare GT, %arg0, %arg2, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %4 = stablehlo.compare LE, %arg1, %arg2, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %5 = stablehlo.compare LT, %arg1, %arg3, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>

  %s0 = stablehlo.select %0, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s1 = stablehlo.select %1, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s2 = stablehlo.select %2, %arg0, %arg1 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s3 = stablehlo.select %3, %arg0, %arg2 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s4 = stablehlo.select %4, %arg1, %arg2 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %s5 = stablehlo.select %5, %arg1, %arg3 : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>

  // CHECK-DAG:  [[C0:%.+]] = stablehlo.compare EQ, [[ARG0]], [[ARG1]], SIGNED
  // CHECK-DAG:  [[C1:%.+]] = stablehlo.compare NE, [[ARG0]], [[ARG1]], SIGNED

  // CHECK-DAG:  [[S0:%.+]] = stablehlo.select [[C0]], [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[S1:%.+]] = stablehlo.select [[C1]], [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[S2:%.+]] = stablehlo.maximum [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[S3:%.+]] = stablehlo.maximum [[ARG0]], [[ARG2]]
  // CHECK-DAG:  [[S4:%.+]] = stablehlo.minimum [[ARG1]], [[ARG2]]
  // CHECK-DAG:  [[S5:%.+]] = stablehlo.minimum [[ARG1]], [[ARG3]]

  // CHECK-NEXT: return [[S0]], [[S1]], [[S2]], [[S3]], [[S4]], [[S5]]
  return %s0, %s1, %s2, %s3, %s4, %s5 :
         tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL: func.func @select_into_minmax2
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<i32>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tensor<i32>, [[ARG3:%.+]]: tensor<i32>)
func.func @select_into_minmax2(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>)
  -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>,
      tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {

  %0 = stablehlo.compare GT, %arg1, %arg0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = stablehlo.compare GT, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = stablehlo.compare GE, %arg1, %arg3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %3 = stablehlo.compare GE, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %s0 = stablehlo.select %0, %arg0, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s1 = stablehlo.select %1, %arg0, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s2 = stablehlo.select %2, %arg3, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s3 = stablehlo.select %3, %arg0, %arg2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>

  %4 = stablehlo.compare LT, %arg1, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %5 = stablehlo.compare LT, %arg0, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %6 = stablehlo.compare LE, %arg2, %arg3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %7 = stablehlo.compare LE, %arg0, %arg2, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>

  %s4 = stablehlo.select %4, %arg2, %arg1 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s5 = stablehlo.select %5, %arg1, %arg2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s6 = stablehlo.select %6, %arg3, %arg2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %s7 = stablehlo.select %7, %arg2, %arg3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>

  // CHECK-DAG:  [[C1:%.+]] = stablehlo.compare GT, [[ARG1]], [[ARG2]], SIGNED
  // CHECK-DAG:  [[C3:%.+]] = stablehlo.compare GE, [[ARG1]], [[ARG2]], SIGNED

  // CHECK-DAG:  [[S0:%.+]] = stablehlo.minimum [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[S1:%.+]] = stablehlo.select [[C1]], [[ARG0]], [[ARG1]]
  // CHECK-DAG:  [[S2:%.+]] = stablehlo.minimum [[ARG3]], [[ARG1]]
  // CHECK-DAG:  [[S3:%.+]] = stablehlo.select [[C3]], [[ARG0]], [[ARG2]]

  // CHECK-DAG:  [[C5:%.+]] = stablehlo.compare LT, [[ARG0]], [[ARG2]], SIGNED
  // CHECK-DAG:  [[C7:%.+]] = stablehlo.compare LE, [[ARG0]], [[ARG2]], SIGNED

  // CHECK-DAG:  [[S4:%.+]] = stablehlo.maximum [[ARG2]], [[ARG1]]
  // CHECK-DAG:  [[S5:%.+]] = stablehlo.select [[C5]], [[ARG1]], [[ARG2]]
  // CHECK-DAG:  [[S6:%.+]] = stablehlo.maximum [[ARG3]], [[ARG2]]
  // CHECK-DAG:  [[S7:%.+]] = stablehlo.select [[C7]], [[ARG2]], [[ARG3]]

  // CHECK-NEXT: return [[S0]], [[S1]], [[S2]], [[S3]], [[S4]], [[S5]], [[S6]], [[S7]]
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>,
                                                  tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

/////////
// TransposeOp

// CHECK-LABEL: @transpose_identity
func.func @transpose_identity(%arg0: tensor<2xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<f32>)
          -> (tensor<2xf32>, tensor<3x2xf32>, tensor<2x3xf32>, tensor<f32>) {
  %a = stablehlo.transpose %arg0, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
  %b = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<3x2xf32>) -> tensor<3x2xf32>
  %c = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
  %d = stablehlo.transpose %arg2, dims = [] : (tensor<f32>) -> tensor<f32>

  // CHECK-NEXT: [[X:%.+]] = stablehlo.transpose %arg1, dims = [1, 0]
  // CHECK-NEXT: return %arg0, %arg1, [[X]], %arg2
  return %a, %b, %c, %d : tensor<2xf32>, tensor<3x2xf32>, tensor<2x3xf32>, tensor<f32>
}

// CHECK-LABEL: @transpose_is_reshape
func.func @transpose_is_reshape(%arg0: tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32> {
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  return %0 : tensor<1x4x1x5xf32>
}

// CHECK-LABEL: @transpose_is_not_reshape
func.func @transpose_is_not_reshape(%arg0: tensor<1x4x5x2xf32>) -> tensor<2x4x1x5xf32> {
  // CHECK-NOT: stablehlo.reshape
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<1x4x5x2xf32>) -> tensor<2x4x1x5xf32>
  return %0 : tensor<2x4x1x5xf32>
}

// -----

/////////
// Generic Zero Extent Ops

// CHECK-LABEL: func.func @reduce_zero_ext
func.func @reduce_zero_ext(%arg0: tensor<0xi1>) -> tensor<i32> {
  %0 = stablehlo.constant dense<false> : tensor<i1>
  %1 = stablehlo.constant dense<false> : tensor<0xi1>
  %2 = stablehlo.compare NE, %arg0, %1, UNSIGNED : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>
  %3 = stablehlo.convert %2 : (tensor<0xi1>) -> tensor<0xi32>
  %4 = stablehlo.constant dense<0> : tensor<i32>
  %5 = stablehlo.reduce(%3 init: %4) across dimensions = [0] : (tensor<0xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg1: tensor<i32>, %arg2: tensor<i32>)  {
    %6 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %6 : tensor<i32>
  }

  // CHECK: [[CST:%.+]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: return [[CST]] : tensor<i32>
  return %5 : tensor<i32>
}

// -----

/////////
// XorOp

// CHECK-LABEL: @xor_cst_on_rhs
func.func @xor_cst_on_rhs(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %cst = stablehlo.constant dense<false> : tensor<2xi1>
  %0 = stablehlo.xor %cst, %arg0 : tensor<2xi1>
  // CHECK: stablehlo.xor %arg0, %c : tensor<2xi1>
  return %0 : tensor<2xi1>
}

// -----

/////////
// Zero Extents

// CHECK-LABEL: func.func @add_zero_ext
func.func @add_zero_ext(%arg0 : tensor<5x0xi32>, %arg1 : tensor<5x0xi32>) -> tensor<5x0xi32> {
  // CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<5x0xi32>
  // CHECK:   return %[[EMPTY]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<5x0xi32>
  func.return %0 : tensor<5x0xi32>
}

// -----

// CHECK-LABEL: func.func @add_zero_ext_dynamic
func.func @add_zero_ext_dynamic(%arg0 : tensor<?x0xi32>, %arg1 : tensor<?x0xi32>) -> tensor<?x0xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x0xi32>
  // CHECK-NOT:   tensor.empty()
  func.return %0 : tensor<?x0xi32>
}

// -----

// CHECK-LABEL: func.func @scatter_zero_ext
func.func @scatter_zero_ext(%arg0 : tensor<f32>, %arg1 : tensor<1x0xi32>, %arg2 : tensor<1xf32>) -> tensor<f32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<f32>, tensor<1x0xi32>, tensor<1xf32>) -> tensor<f32>
  // CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x0xi32>
  // CHECK:   %[[SCATTER:.+]] = "stablehlo.scatter"(%arg0, %0, %arg2)
  // CHECK:   return %[[SCATTER]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @sort_zero_extent
func.func public @sort_zero_extent(%arg0: tensor<0xi16> {jax.arg_info = "a", mhlo.sharding = "{replicated}"}) -> (tensor<0xi32> {jax.result_info = ""}) {
  %0 = stablehlo.iota dim = 0 : tensor<0xi32>
  %1:2 = "stablehlo.sort"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i16>, %arg2: tensor<i16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %2 = stablehlo.compare  LT, %arg1, %arg2,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  }) {dimension = 0 : i64, is_stable = true} : (tensor<0xi16>, tensor<0xi32>) -> (tensor<0xi16>, tensor<0xi32>)

  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<0xi32>
  // CHECK: return %[[EMPTY]]
  return %1#1 : tensor<0xi32>
}


// -----

// CHECK-LABEL: @while_zero_extent
// CHECK: %[[R0:.+]] = tensor.empty() : tensor<75x0xf32>
// CHECK: %[[R1:.+]] = tensor.empty() : tensor<75x0xf32>
// CHECK: %[[R2:.+]]:2 = stablehlo.while
// CHECK: return %[[R2]]#0, %[[R0]]


func.func public @while_zero_extent(%arg0: tensor<i32>, %arg1: tensor<3xf32>, %arg2: tensor<75x0xf32>) -> (tensor<i32>, tensor<75x0xf32>) {
  %0 = stablehlo.constant dense<1> : tensor<i32>
  %1 = stablehlo.constant dense<75> : tensor<i32>
  %2 = stablehlo.constant dense<0> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %2, %iterArg_2 = %arg2) : tensor<i32>, tensor<75x0xf32>
   cond {
    %4 = stablehlo.compare  LT, %iterArg, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %44 = stablehlo.add %iterArg, %0 : tensor<i32>
    stablehlo.return %44, %iterArg_2 : tensor<i32>, tensor<75x0xf32>
  }
  return %3#0, %3#1 : tensor<i32>, tensor<75x0xf32>
}

// -----

/////////
// Generic Shape Ops

// CHECK-LABEL: @push_shape_ops_to_end
func.func @push_shape_ops_to_end(%arg0 : tensor<12xf32>) -> tensor<3x4x2x1xf32> {
  // CHECK: %[[COS:.+]] = stablehlo.cosine %arg0 : tensor<12xf32>
  // CHECK: %[[ABS:.+]] = stablehlo.abs %[[COS]] : tensor<12xf32>
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %[[ABS]] : (tensor<12xf32>) -> tensor<3x4xf32>
  // CHECK: %[[BROADCAST:.+]] = stablehlo.broadcast %[[RESHAPE]], sizes = [1, 2] : (tensor<3x4xf32>) -> tensor<1x2x3x4xf32>
  // CHECK: %[[TRANSPOSE:.+]] = stablehlo.transpose %[[BROADCAST]], dims = [2, 3, 1, 0] : (tensor<1x2x3x4xf32>) -> tensor<3x4x2x1xf32>
  // CHECK: return %[[TRANSPOSE]]
  %0 = stablehlo.reshape %arg0 : (tensor<12xf32>) -> tensor<3x4xf32>
  %1 = stablehlo.broadcast %0, sizes = [1, 2] : (tensor<3x4xf32>) -> tensor<1x2x3x4xf32>
  %2 = stablehlo.cosine %1 : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %3 = stablehlo.transpose %2, dims = [2, 3, 1, 0]  : (tensor<1x2x3x4xf32>) -> tensor<3x4x2x1xf32>
  %4 = stablehlo.abs %3 : (tensor<3x4x2x1xf32>) -> tensor<3x4x2x1xf32>
  return %4 : tensor<3x4x2x1xf32>
}


// -----

// CHECK-LABEL: @reorder_with_type_change
func.func @reorder_with_type_change(%arg0 : tensor<3x4xi32>) -> tensor<12xi64> {
  // CHECK: %[[CONVERT:.+]] = stablehlo.convert %arg0 : (tensor<3x4xi32>) -> tensor<3x4xi64>
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %[[CONVERT]] : (tensor<3x4xi64>) -> tensor<12xi64>
  // CHECK: return %[[RESHAPE]]
  %0 = stablehlo.reshape %arg0 : (tensor<3x4xi32>) -> tensor<12xi32>
  %1 = stablehlo.convert %0 : (tensor<12xi32>) -> tensor<12xi64>
  return %1 : tensor<12xi64>
}


// -----

// CHECK-LABEL: @do_not_reorder_with_other_uses
func.func @do_not_reorder_with_other_uses(%arg0: tensor<2x2xf64>, %arg1: tensor<4xf32>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<4xf32>) {
  // CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<2x2xf64>) -> tensor<4xf64>
  // CHECK: %[[CONVERT:.+]] = stablehlo.convert %[[RESHAPE]] : (tensor<4xf64>) -> tensor<4xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<2x2xf64>) -> tensor<4xf64>
  %1 = stablehlo.convert %0 : (tensor<4xf64>) -> tensor<4xf32>
  %2 = stablehlo.subtract %arg1, %1 : tensor<4xf32>
  %3 = stablehlo.reduce(%0 init: %arg2) across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    reducer(%arg3: tensor<f64>, %arg4: tensor<f64>)  {
    %4 = stablehlo.add %arg3, %arg4 : tensor<f64>
    stablehlo.return %4 : tensor<f64>
  }
  return %3, %2 : tensor<f64>, tensor<4xf32>
}


// -----

// Make sure we do not crash on unregistered dialects.

// CHECK-LABEL: func.func @generic_op
func.func @generic_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT:    "test_dialect.op"
  // CHECK-NEXT:    return
  %0 = "test_dialect.op"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>)
  return %0 : tensor<2xf32>
}
