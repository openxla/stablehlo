// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @round_nearest_even_op_test_f64() {
  %operand = stablehlo.constant dense<[-2.5, 0.4, 0.5, 0.6, 2.5]> : tensor<5xf64>
  %result = stablehlo.round_nearest_even %operand : tensor<5xf64>
  check.expect_almost_eq_const %result, dense<[-2.0, 0.0, 0.0, 1.0, 2.0]> : tensor<5xf64>
  func.return
}
