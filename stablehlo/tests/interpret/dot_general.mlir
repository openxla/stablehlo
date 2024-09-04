// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dot_general_op_test_si64() {
  %lhs = stablehlo.constant dense<[[[1, 2], [3, 4]],
                                   [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %rhs = stablehlo.constant dense<[[[1, 0], [0, 1]],
                                   [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %result = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0] x [0],
    contracting_dims = [2] x [1],
    precision = [DEFAULT, DEFAULT]
    : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
  check.expect_eq_const %result, dense<[[[1, 2], [3, 4]],
                                        [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  func.return
}

// -----

func.func @dot_general_op_test_algorithm() {
  %lhs = stablehlo.constant dense<[[[1, 2], [3, 4]],
                                   [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  %rhs = stablehlo.constant dense<[[[1, 0], [0, 1]],
                                   [[1, 0], [0, 1]]]> : tensor<2x2x2xi64>
  %result = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0] x [0],
    contracting_dims = [2] x [1],
    precision = [DEFAULT, DEFAULT],
    algorithm = <
      lhs_precision_type = tf32,
      rhs_precision_type = tf32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    > : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
  check.expect_eq_const %result, dense<[[[1, 2], [3, 4]],
                                        [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>
  func.return
}

// -----

func.func @dot_general_op_test_empty_dims() {
  %lhs = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
  %rhs = stablehlo.constant dense<[[1, 0], [0, 1]]> : tensor<2x2xi64>
  %result = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [] x [],
    contracting_dims = [] x [],
    precision = [DEFAULT, DEFAULT]
    : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2x2x2xi64>
  check.expect_eq_const %result, dense<[[[[1, 0], [0, 1]],
                                         [[2, 0], [0, 2]]],
                                        [[[3, 0], [0, 3]],
                                         [[4, 0], [0, 4]]]]> : tensor<2x2x2x2xi64>
  func.return
}

// -----

func.func @dot_general_op_test_different_operand_and_result_element_types() {
  %lhs = stablehlo.constant dense<[[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>
  %rhs = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0]],
                                  [[1.0, 0.0], [0.0, 1.0]]]> : tensor<2x2x2xf32>
  %result = stablehlo.dot_general %lhs, %rhs,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf64>
  check.expect_eq_const %result, dense<[[[1.0, 2.0], [3.0, 4.0]],
                                        [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf64>
  func.return
}

// -----

func.func @add_op_test_f8E3M4() {
  %0 = stablehlo.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf8E3M4>
  %result = stablehlo.dot_general %0, %0,
    contracting_dims = [0] x [0]
    : (tensor<4xf8E3M4>, tensor<4xf8E3M4>) -> tensor<f8E3M4>
  check.expect_almost_eq_const %result, dense<14.0> : tensor<f8E3M4>
  func.return
}

// -----

func.func @add_op_test_f8E4M3() {
  %0 = stablehlo.constant dense<[0.0, 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0, 7.0]> : tensor<8xf8E4M3>
  %result = stablehlo.dot_general %0, %0,
    contracting_dims = [0] x [0]
    : (tensor<8xf8E4M3>, tensor<8xf8E4M3>) -> tensor<f8E4M3>
  check.expect_almost_eq_const %result, dense<140.0> : tensor<f8E4M3>
  func.return
}
