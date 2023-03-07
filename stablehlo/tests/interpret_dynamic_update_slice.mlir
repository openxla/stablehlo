// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dynamic_update_slice() {
  %operand = stablehlo.constant dense<[[1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       [1, 2, 2, 2],
                                       [1, 2, 2, 2]]> : tensor<4x4xi64>
  %update = stablehlo.constant dense<[[1, 1, 1],
                                      [1, 1, 1]]> : tensor<2x3xi64>
  %start_indices0 = stablehlo.constant dense<4> : tensor<i64>
  %start_indices1 = stablehlo.constant dense<4> : tensor<i64>
  %result = stablehlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1 :
      (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
  check.expect_eq_const %result, dense<[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]> : tensor<4x4xi64>
  func.return
}
