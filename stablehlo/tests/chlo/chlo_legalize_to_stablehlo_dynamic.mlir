// RUN: stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @constant_like_dynamic
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:       %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[SHAPE:.*]] = shape.shape_of %[[ARG0]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[RES:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CST]], %[[SHAPE]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:           return %[[RES]] : tensor<?x?xf32>
func.func @constant_like_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "chlo.constant_like"(%arg0) { value = 1.0 : f32 } : (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @top_k_dynamic
// CHECK-SAME:      %[[ARG0:.*]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
// CHECK-DAG:       %[[GET_DIM0:.*]] = stablehlo.get_dimension_size %[[ARG0]], dim = 0 : (tensor<?x?xf32>) -> tensor<i32>
// CHECK-DAG:       %[[CONVERT0:.*]] = stablehlo.convert %[[GET_DIM0]] : (tensor<i32>) -> tensor<i64>
// CHECK-DAG:       %[[RESHAPE0:.*]] = stablehlo.reshape %[[CONVERT0]] : (tensor<i64>) -> tensor<1xi64>
// CHECK-DAG:       %[[GET_DIM1:.*]] = stablehlo.get_dimension_size %[[ARG0]], dim = 1 : (tensor<?x?xf32>) -> tensor<i32>
// CHECK-DAG:       %[[CONVERT1:.*]] = stablehlo.convert %[[GET_DIM1]] : (tensor<i32>) -> tensor<i64>
// CHECK-DAG:       %[[RESHAPE1:.*]] = stablehlo.reshape %[[CONVERT1]] : (tensor<i64>) -> tensor<1xi64>
// CHECK:           %[[CONCAT_SHAPE:.*]] = stablehlo.concatenate %[[RESHAPE0]], %[[RESHAPE1]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       %[[C5:.*]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK-DAG:       %[[RESHAPE_C5:.*]] = stablehlo.reshape %[[C5]] : (tensor<i64>) -> tensor<1xi64>
// CHECK-DAG:       %[[LIMIT:.*]] = stablehlo.concatenate %[[RESHAPE0]], %[[RESHAPE_C5]], dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           %[[IOTA:.*]] = stablehlo.dynamic_iota %[[CONCAT_SHAPE]], dim = 1 : (tensor<2xi64>) -> tensor<?x?xi32>
// CHECK:           %[[SORT:.*]]:2 = "stablehlo.sort"(%[[ARG0]], %[[IOTA]])
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK:           %[[C1:.*]] = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK:           %[[SLICE_VAL:.*]] = stablehlo.real_dynamic_slice %[[SORT]]#0, %[[C0]], %[[LIMIT]], %[[C1]] : (tensor<?x?xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:           %[[SLICE_IDX:.*]] = stablehlo.real_dynamic_slice %[[SORT]]#1, %[[C0]], %[[LIMIT]], %[[C1]] : (tensor<?x?xi32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xi32>
// CHECK:           return %[[SLICE_VAL]], %[[SLICE_IDX]] : tensor<?x?xf32>, tensor<?x?xi32>
func.func @top_k_dynamic(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %0:2 = chlo.top_k(%arg0, k=5) : tensor<?x?xf32> -> (tensor<?x?xf32>, tensor<?x?xi32>)
  func.return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func.func @ragged_dot_dynamic
// CHECK-SAME:      %[[LHS:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<3x2x5x7xf32>,
// CHECK-SAME:      %[[GROUP_SIZES:.*]]: tensor<3xi64>) -> tensor<?x?x?xf32> {
// CHECK:           %[[C0_I64:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:       %[[C1_I64:.*]] = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:       %[[DIM1_LHS:.*]] = stablehlo.get_dimension_size %[[LHS]], dim = 1 : (tensor<?x?x?xf32>) -> tensor<i32>
// CHECK-DAG:       %[[CONVERT_DIM1:.*]] = stablehlo.convert %[[DIM1_LHS]] : (tensor<i32>) -> tensor<i64>
// CHECK-DAG:       %[[RESHAPE_DIM1:.*]] = stablehlo.reshape %[[CONVERT_DIM1]] : (tensor<i64>) -> tensor<1xi64>
// CHECK:           %[[SHAPE_I64:.*]] = stablehlo.concatenate %{{.*}}, %[[RESHAPE_DIM1]], %{{.*}}, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           %[[IOTA:.*]] = stablehlo.dynamic_iota %[[SHAPE_I64]], dim = 1 : (tensor<3xi64>) -> tensor<1x?x1xi64>
// CHECK:           %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[SHAPE_LHS:.*]] = shape.shape_of %[[LHS]] : tensor<?x?x?xf32> -> tensor<3xindex>
// CHECK:           %[[BROADCAST_ZERO:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ZERO]], %{{.*}}, dims = [] : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK:           %[[SLICE_G0:.*]] = stablehlo.slice %[[GROUP_SIZES]] [0:1] : (tensor<3xi64>) -> tensor<1xi64>
// CHECK:           %[[RESHAPE_G0:.*]] = stablehlo.reshape %[[SLICE_G0]] : (tensor<1xi64>) -> tensor<i64>
// CHECK:           %[[ADD_LIMIT:.*]] = stablehlo.add %[[C0_I64]], %[[RESHAPE_G0]] : tensor<i64>
// CHECK:           %[[BROADCAST_START:.*]] = stablehlo.dynamic_broadcast_in_dim %[[C0_I64]], %{{.*}}, dims = [] : (tensor<i64>, tensor<3xindex>) -> tensor<?x?x?xi64>
// CHECK:           %[[BROADCAST_LIMIT:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ADD_LIMIT]], %{{.*}}, dims = [] : (tensor<i64>, tensor<3xindex>) -> tensor<?x?x?xi64>
// CHECK:           %[[BROADCAST_IOTA:.*]] = stablehlo.dynamic_broadcast_in_dim %[[IOTA]], %{{.*}}, dims = [0, 1, 2] : (tensor<1x?x1xi64>, tensor<3xindex>) -> tensor<?x?x?xi64>
// CHECK:           %[[COMPARE_START:.*]] = stablehlo.compare  GE, %[[BROADCAST_IOTA]], %[[BROADCAST_START]] : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi1>
// CHECK:           %[[COMPARE_LIMIT:.*]] = stablehlo.compare  LT, %[[BROADCAST_IOTA]], %[[BROADCAST_LIMIT]] : (tensor<?x?x?xi64>, tensor<?x?x?xi64>) -> tensor<?x?x?xi1>
// CHECK:           %[[AND:.*]] = stablehlo.and %[[COMPARE_START]], %[[COMPARE_LIMIT]] : tensor<?x?x?xi1>
// CHECK:           %[[SELECT:.*]] = stablehlo.select %[[AND]], %[[LHS]], %[[BROADCAST_ZERO]] : tensor<?x?x?xi1>, tensor<?x?x?xf32>
// CHECK:           %[[SLICE_RHS0:.*]] = stablehlo.slice %[[RHS]] [0:1, 0:2, 0:5, 0:7] : (tensor<3x2x5x7xf32>) -> tensor<1x2x5x7xf32>
// CHECK:           %[[RESHAPE_RHS0:.*]] = stablehlo.reshape %[[SLICE_RHS0]] : (tensor<1x2x5x7xf32>) -> tensor<2x5x7xf32>
// CHECK:           %[[DOT0:.*]] = stablehlo.dot_general %[[SELECT]], %[[RESHAPE_RHS0]]
func.func @ragged_dot_dynamic(%lhs : tensor<?x?x?xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<?x?x?xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
