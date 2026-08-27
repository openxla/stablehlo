// RUN: stablehlo-opt %s --hlo-test-broadcast --split-input-file --allow-unregistered-dialect | FileCheck %s

/////////
// Scalar broadcast tests.

// [] x [1] => [1]
// CHECK-LABEL: func @scalar_broadcast_scalar_x_1
func.func @scalar_broadcast_scalar_x_1(%arg0: tensor<f64>, %arg1: tensor<1xf64>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<1xf64>
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST]], %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<f64>, tensor<1xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [1] x [] => [1]
// CHECK-LABEL: func @scalar_broadcast_1_x_scalar
func.func @scalar_broadcast_1_x_scalar(%arg0: tensor<1xf64>, %arg1: tensor<f64>) -> !stablehlo.token {
  // CHECK: %[[RHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<1xf64>
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %[[RHS_BCAST]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<1xf64>, tensor<f64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [] x [10] => [10]
// CHECK-LABEL: func @scalar_broadcast_scalar_x_10
func.func @scalar_broadcast_scalar_x_10(%arg0: tensor<f64>, %arg1: tensor<10xf64>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<10xf64>
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST]], %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<f64>, tensor<10xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [<=10] x [] => [<=10]
// CHECK-LABEL: func @scalar_broadcast_b10_x_scalar
func.func @scalar_broadcast_b10_x_scalar(%arg0: tensor<?xf64, #stablehlo.bounds<10>>, %arg1: tensor<f64>) -> !stablehlo.token {
  // CHECK: %[[RHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<10xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 0
  // CHECK: %[[RHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[RHS_BCAST]], %[[DIM_SIZE]], dim = 0
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %[[RHS_BCAST_DYN]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<?xf64, #stablehlo.bounds<10>>, tensor<f64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [] x [<=10] => [<=10]
// CHECK-LABEL: func @scalar_broadcast_scalar_x_b10
func.func @scalar_broadcast_scalar_x_b10(%arg0: tensor<f64>, %arg1: tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<10xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg1, dim = 0
  // CHECK: %[[LHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[LHS_BCAST]], %[[DIM_SIZE]], dim = 0
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST_DYN]], %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<f64>, tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [] x [1, <=10, 1] => [1, <=10, 1]
// CHECK-LABEL: func @scalar_broadcast_scalar_x_1_b10_1
func.func @scalar_broadcast_scalar_x_1_b10_1(%arg0: tensor<f64>, %arg1: tensor<1x?x1xf64, #stablehlo.bounds<?, 10, ?>>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<1x10x1xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg1, dim = 1
  // CHECK: %[[LHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[LHS_BCAST]], %[[DIM_SIZE]], dim = 1
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST_DYN]], %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<f64>, tensor<1x?x1xf64, #stablehlo.bounds<?, 10, ?>>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// [10, 1, <=5] x [] => [10, 1, <=5]
// CHECK-LABEL: func @scalar_broadcast_10_1_b5_x_scalar
func.func @scalar_broadcast_10_1_b5_x_scalar(%arg0: tensor<10x1x?xf64, #stablehlo.bounds<?, ?, 5>>, %arg1: tensor<f64>) -> !stablehlo.token {
  // CHECK: %[[RHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<10x1x5xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 2
  // CHECK: %[[RHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[RHS_BCAST]], %[[DIM_SIZE]], dim = 2
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %[[RHS_BCAST_DYN]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<10x1x?xf64, #stablehlo.bounds<?, ?, 5>>, tensor<f64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

//////
// 1-D SCALAR TESTS

// [1] x [1] => [1]
// [1] x [10] => [1]
// [<=10] x [1] => [<=10]
// [1] x [<=10] => [<=10]
// [1] x [1, <=10, 1] => [1, <=10, 1]
// [5] x [10, 1] => [10, 5]
// [5] x [<=10, 1] => [<=10, 5]


// [1] x [1] => [1]
// CHECK-LABEL: func @single_dim_scalar_1_x_1
func.func @single_dim_scalar_1_x_1(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>) -> !stablehlo.token {
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<1xf64>, tensor<1xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [1] x [10] => [10]
// CHECK-LABEL: func @single_dim_scalar_1_x_10
func.func @single_dim_scalar_1_x_10(%arg0: tensor<1xf64>, %arg1: tensor<10xf64>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf64>) -> tensor<10xf64>
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST]], %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<1xf64>, tensor<10xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [<=10] x [1] => [<=10]
// CHECK-LABEL: func @single_dim_scalar_b10_x_1
func.func @single_dim_scalar_b10_x_1(%arg0: tensor<?xf64, #stablehlo.bounds<10>>, %arg1: tensor<1xf64>) -> !stablehlo.token {
  // CHECK: %[[RHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xf64>) -> tensor<10xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 0
  // CHECK: %[[RHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[RHS_BCAST]], %[[DIM_SIZE]], dim = 0
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %[[RHS_BCAST_DYN]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<?xf64, #stablehlo.bounds<10>>, tensor<1xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [1] x [<=10] => [<=10]
// CHECK-LABEL: func @single_dim_scalar_1_x_b10
func.func @single_dim_scalar_1_x_b10(%arg0: tensor<1xf64>, %arg1: tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf64>) -> tensor<10xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg1, dim = 0
  // CHECK: %[[LHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[LHS_BCAST]], %[[DIM_SIZE]], dim = 0
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST_DYN]], %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<1xf64>, tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// [<=10] x [<=10] => [<=10] // PT layer must ensure these are identical!
// CHECK-LABEL: func @single_dim_scalar_b10_x_b10
func.func @single_dim_scalar_b10_x_b10(%arg0: tensor<?xf64, #stablehlo.bounds<10>>, %arg1: tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token {
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<?xf64, #stablehlo.bounds<10>>, tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [1] x [1, <=10, 1] => [1, <=10, 1]
// CHECK-LABEL: func @single_dim_scalar_1_x_1_b10_1
func.func @single_dim_scalar_1_x_1_b10_1(%arg0: tensor<1xf64>, %arg1: tensor<1x?x1xf64, #stablehlo.bounds<?, 10, ?>>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<1xf64>) -> tensor<1x10x1xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg1, dim = 1
  // CHECK: %[[LHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[LHS_BCAST]], %[[DIM_SIZE]], dim = 1
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST_DYN]], %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<1xf64>, tensor<1x?x1xf64, #stablehlo.bounds<?, 10, ?>>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [10, 1, <=5] x [1] => [10, 1, <=5]
// CHECK-LABEL: func @single_dim_scalar_10_1_b5_x_1
func.func @single_dim_scalar_10_1_b5_x_1(%arg0: tensor<10x1x?xf64, #stablehlo.bounds<?, ?, 5>>, %arg1: tensor<1xf64>) -> !stablehlo.token {
  // CHECK: %[[RHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<1xf64>) -> tensor<10x1x5xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 2
  // CHECK: %[[RHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[RHS_BCAST]], %[[DIM_SIZE]], dim = 2
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %[[RHS_BCAST_DYN]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<10x1x?xf64, #stablehlo.bounds<?, ?, 5>>, tensor<1xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}


//////
// N-D Tests

// [1, 2] x [1, 2] => [1, 2]
// CHECK-LABEL: func @tensor_no_broadcast_match
func.func @tensor_no_broadcast_match(%arg0: tensor<1x2xf64>, %arg1: tensor<1x2xf64>) -> !stablehlo.token {
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %arg1)
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<1x2xf64>, tensor<1x2xf64>) ->  !stablehlo.token
  return %0 : !stablehlo.token
}

// [10, 1] x [1, 1] => [10, 1]
// CHECK-LABEL: func @tensor_broadcast_10_1_x_1_1
func.func @tensor_broadcast_10_1_x_1_1(%arg0: tensor<10x1xf64>, %arg1: tensor<1x1xf64>) -> !stablehlo.token {
  // CHECK: %[[RHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x1xf64>) -> tensor<10x1xf64>
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%arg0, %[[RHS_BCAST]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<10x1xf64>, tensor<1x1xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [<=10, 1] x [1, 10] => [<=10, 10]
// CHECK-LABEL: func @tensor_broadcast_b10_1_x_1_10
func.func @tensor_broadcast_b10_1_x_1_10(%arg0: tensor<?x1xf64, #stablehlo.bounds<10, ?>>, %arg1: tensor<1x10xf64>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<?x1xf64, #stablehlo.bounds<10, ?>>) -> tensor<?x10xf64, #stablehlo.bounds<10, ?>>
  // CHECK: %[[RHS_BCAST_STATIC:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x10xf64>) -> tensor<10x10xf64>
  // CHECK: %[[DIM_SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 0
  // CHECK: %[[RHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[RHS_BCAST_STATIC]], %[[DIM_SIZE]], dim = 0
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST]], %[[RHS_BCAST_DYN]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<?x1xf64, #stablehlo.bounds<10, ?>>, tensor<1x10xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [<=10, 1] x [1, <=10] => [<=10, <=10]
// CHECK-LABEL: func @tensor_broadcast_b10_1_x_1_b10
func.func @tensor_broadcast_b10_1_x_1_b10(
  %arg0: tensor<?x1xf64, #stablehlo.bounds<10, ?>>,
  %arg1: tensor<1x?xf64, #stablehlo.bounds<?, 10>>
) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST_STATIC:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<?x1xf64, #stablehlo.bounds<10, ?>>) -> tensor<?x10xf64, #stablehlo.bounds<10, ?>>
  // CHECK: %[[ARG1_DIM1_SIZE:.+]] = stablehlo.get_dimension_size %arg1, dim = 1
  // CHECK: %[[LHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[LHS_BCAST_STATIC]], %[[ARG1_DIM1_SIZE]], dim = 1
  // CHECK: %[[RHS_BCAST_STATIC:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<1x?xf64, #stablehlo.bounds<?, 10>>) -> tensor<10x?xf64, #stablehlo.bounds<?, 10>>
  // CHECK: %[[ARG0_DIM0_SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 0
  // CHECK: %[[RHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[RHS_BCAST_STATIC]], %[[ARG0_DIM0_SIZE]], dim = 0
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST_DYN]], %[[RHS_BCAST_DYN]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (
    tensor<?x1xf64, #stablehlo.bounds<10, ?>>,
    tensor<1x?xf64, #stablehlo.bounds<?, 10>>
  ) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [5] x [10, 1] => [10, 5]
// CHECK-LABEL: func @tensor_broadcast_5_x_10_1
func.func @tensor_broadcast_5_x_10_1(%arg0: tensor<5xf64>, %arg1: tensor<10x1xf64>) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<5xf64>) -> tensor<10x5xf64>
  // CHECK: %[[RHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<10x1xf64>) -> tensor<10x5xf64>
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST]], %[[RHS_BCAST]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<5xf64>, tensor<10x1xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [<=10, 1] x [5] => [<=10, 5]
// CHECK-LABEL: func @tensor_broadcast_b5_1_x_5
func.func @tensor_broadcast_b5_1_x_5(
  %arg0: tensor<?x1xf64, #stablehlo.bounds<10, ?>>,
  %arg1: tensor<5xf64>
) -> !stablehlo.token {
  // CHECK: %[[LHS_BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<?x1xf64, #stablehlo.bounds<10, ?>>) -> tensor<?x5xf64, #stablehlo.bounds<10, ?>>
  // CHECK: %[[RHS_BCAST_STATIC:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<5xf64>) -> tensor<10x5xf64>
  // CHECK: %[[ARG0_DIM0_SIZE:.+]] = stablehlo.get_dimension_size %arg0, dim = 0
  // CHECK: %[[RHS_BCAST_DYN:.+]] = stablehlo.set_dimension_size %[[RHS_BCAST_STATIC]], %[[ARG0_DIM0_SIZE]], dim = 0
  // CHECK-NEXT: stablehlo.custom_call @numpy_broadcasted(%[[LHS_BCAST]], %[[RHS_BCAST_DYN]])
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (
    tensor<?x1xf64, #stablehlo.bounds<10, ?>>,
    tensor<5xf64>
  ) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

//////
// N-ary broadcast tests.


// [<=10, 1] x [1, <=10] x [1] => [<=10, <=10]
// CHECK-LABEL: func @nary_broadcast_b10_1_x_1_b10_x_1
func.func @nary_broadcast_b10_1_x_1_b10_x_1(
  %arg0: tensor<?x1xf64, #stablehlo.bounds<10, ?>>,
  %arg1: tensor<1x?xf64, #stablehlo.bounds<?, 10>>,
  %arg2: tensor<1xf64>
) -> !stablehlo.token {
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1, %arg2) : (tensor<?x1xf64, #stablehlo.bounds<10, ?>>, tensor<1x?xf64, #stablehlo.bounds<?, 10>>, tensor<1xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

/////
// Broadcast errors

// [10] x [5] => error
// expected-error @+1 {{incompatible shapes for broadcasting 10 and 5}}
func.func @broadcast_error_10_x_5(%arg0: tensor<10xf64>, %arg1: tensor<5xf64>) -> !stablehlo.token {
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<10xf64>, tensor<5xf64>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [10] x [<=10] => error
// expected-error @+1 {{cannot mix bounded and static dimensions in broadcast}}
func.func @broadcast_error_10_x_b10(%arg0: tensor<10xf64>, %arg1: tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token {
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<10xf64>, tensor<?xf64, #stablehlo.bounds<10>>) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [10] x not_tensor => error
func.func @broadcast_error_not_tensor(%arg0: tensor<10xf64>, %arg1: !stablehlo.token) -> !stablehlo.token {
  // expected-error @+1 {{expected ranked tensor type for broadcast inputs}}
  %0 = "hlo_test_broadcast.numpy_broadcast"(%arg0, %arg1) : (tensor<10xf64>, !stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// [] => error
func.func @broadcast_error_empty() -> !stablehlo.token {
  // expected-error @+1 {{requires at least one operand to broadcast}}
  %0 = "hlo_test_broadcast.numpy_broadcast"() : () -> !stablehlo.token
  return %0 : !stablehlo.token
}

