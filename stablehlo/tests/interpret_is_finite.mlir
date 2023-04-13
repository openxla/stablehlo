// RUN: stablehlo-translate --interpret -split-input-file %s

// CHECK-LABEL: Evaluated results of function: is_finite_op_test_f64
func.func @is_finite_op_test_f64() {
  // -Inf, +Inf, NaN, -10.0, -0.0, 0.0, 10.0
  %operand = stablehlo.constant dense<[0xFFF0000000000000, 0x7FF0000000000000, 0x7FF8000000000000, -10.0, -0.0, 0.0, 10.0]> : tensor<7xf64>
  %result = stablehlo.is_finite %operand : (tensor<7xf64>) -> tensor<7xi1>
  check.expect_eq_const %result, dense<[false, false, false, true, true, true, true]> : tensor<7xi1>
  func.return
}
