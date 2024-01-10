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
