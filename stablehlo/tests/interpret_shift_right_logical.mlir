// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @shift_right_logical_op_test_si64() {
  %lhs = stablehlo.constant dense<[-1, 0, 8]> : tensor<3xi64>
  %rhs = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi64>
  %result = stablehlo.shift_right_logical %lhs, %rhs : tensor<3xi64>
  check.expect_eq_const %result, dense<[9223372036854775807, 0, 1]> : tensor<3xi64>
  func.return
}
