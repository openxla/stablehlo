// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @popcnt_op_test_si64() {
  %operand = stablehlo.constant dense<[0, 1, 2, 127]> : tensor<4xi64>
  %result = stablehlo.popcnt %operand : tensor<4xi64>
  check.expect_eq_const %result, dense<[0, 1, 1, 7]> : tensor<4xi64>
  func.return
}
