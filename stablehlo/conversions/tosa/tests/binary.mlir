// RUN: stablehlo-opt %s --stablehlo-legalize-to-tosa | FileCheck %s

// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.add
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @and
func.func @and(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: tosa.bitwise_and
  %0 = "stablehlo.and"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %0 : tensor<10xi32>
}

// CHECK-LABEL: @compare_eq
func.func @compare_eq(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  // CHECK: tosa.equal
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @compare_lt
func.func @compare_lt(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  // CHECK: stablehlo.compare
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @compare_ne
func.func @compare_ne(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi1> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.equal %arg0, %arg1
  // CHECK-DAG: %[[VAR1:.*]] = tosa.logical_not %[[VAR0]]
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction NE>} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @concatenate
func.func @concatenate(%arg0 : tensor<3x3xf32>, %arg1 : tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  return %0 : tensor<6x3xf32>
}

// CHECK-LABEL: @divide
func.func @divide(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: tosa.int_div
  %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %0 : tensor<10xi32>
}

// CHECK-LABEL: @dot_vector_vector
func.func @dot_vector_vector(%arg0 : tensor<3xf32>, %arg1 : tensor<3xf32>) -> tensor<f32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 3>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 1>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @dot_vector_matrix
func.func @dot_vector_matrix(%arg0 : tensor<2xf32>, %arg1 : tensor<2x3xf32>) -> tensor<3xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 2>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 2, 3>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: @dot_matrix_vector
func.func @dot_matrix_vector(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 2, 3>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 1>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @dot_matrix_matrix
func.func @dot_matrix_matrix(%arg0 : tensor<2x3xf32>, %arg1 : tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 2, 3>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 4>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: @dot_general_vector_vector
func.func @dot_general_vector_vector(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<f32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 3>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 1>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: @dot_general_vector_matrix
func.func @dot_general_vector_matrix(%arg0: tensor<2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 2>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 2, 3>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: @dot_general_matrix_vector
func.func @dot_general_matrix_vector(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 2, 3>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 1>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @dot_general_matrix_matrix
func.func @dot_general_matrix_matrix(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK-DAG: %[[VAR0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 2, 3>}
  // CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 4>}
  // CHECK-DAG: %[[VAR2:.*]] = tosa.matmul %[[VAR0]], %[[VAR1]]
  // CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: @gather
func.func @gather(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<3x2xi32>) -> tensor<3x2x5xi32> {
  // CHECK: tosa.gather
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 2, 5>
  } : (tensor<3x4x5xi32>, tensor<3x2xi32>) -> tensor<3x2x5xi32>
  return %0 : tensor<3x2x5xi32>
}

// CHECK-LABEL: @maximum
func.func @maximum(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.maximum
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @maximum_f64
func.func @maximum_f64(%arg0 : tensor<10xf64>, %arg1 : tensor<10xf64>) -> tensor<10xf64> {
  // CHECK: tosa.maximum
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<10xf64>, tensor<10xf64>) -> tensor<10xf64>
  return %0 : tensor<10xf64>
}

// CHECK-LABEL: @minimum
func.func @minimum(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.minimum
  %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @multiply
func.func @multiply(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.mul
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @or
func.func @or(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: tosa.bitwise_or
  %0 = "stablehlo.or"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %0 : tensor<10xi32>
}

// CHECK-LABEL: @power
func.func @power(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.pow
  %0 = "stablehlo.power"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @reduce_max
func.func @reduce_max(%arg0: tensor<1x10xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  // CHECK: tosa.reduce_max
  // CHECK: tosa.reshape
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: @reduce_sum
func.func @reduce_sum(%arg0: tensor<5x4xf32>, %arg1: tensor<f32>) -> tensor<4xf32> {
  // CHECK: tosa.reduce_sum
  // CHECK: tosa.reshape
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @shift_left
func.func @shift_left(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: tosa.logical_left_shift
  %0 = "stablehlo.shift_left"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %0 : tensor<10xi32>
}

// CHECK-LABEL: @shift_right_logical
func.func @shift_right_logical(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: tosa.logical_right_shift
  %0 = "stablehlo.shift_right_logical"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %0 : tensor<10xi32>
}

// CHECK-LABEL: @subtract
func.func @subtract(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.sub
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @xor
func.func @xor(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: tosa.bitwise_xor
  %0 = "stablehlo.xor"(%arg0, %arg1) : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %0 : tensor<10xi32>
}
