// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @count_leading_zeros_op_test_si64() {
  %operand = stablehlo.constant dense<[[0, 1], [128, -1]]> : tensor<2x2xi64>
  %result = stablehlo.count_leading_zeros %operand : tensor<2x2xi64>
  check.expect_eq_const %result, dense<[[64, 63], [56, 0]]> : tensor<2x2xi64>
  func.return
}
