// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dynamic_slice() {
  %operand = stablehlo.constant dense<[[1, 1, 1],
                                       [1, 1, 1],
                                       [1, 1, 1]]> : tensor<3x3xi64>
  %start_indices0 = stablehlo.constant dense<3> : tensor<i64>
  %start_indices1 = stablehlo.constant dense<3> : tensor<i64>
  %result = "stablehlo.dynamic_slice"(%operand, %start_indices0, %start_indices1) {
    slice_sizes = dense<[3, 3]> : tensor<2xi64>
  } : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
  check.expect_eq_const %result, dense<[[1, 1, 1], [1, 1, 1], [1, 1, 1]]> : tensor<3x3xi64>
  func.return
}
