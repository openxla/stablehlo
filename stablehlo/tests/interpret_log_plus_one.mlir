// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @log_op_test_i64() {
  %operand = stablehlo.constant dense<[-2.0, -0.0, -0.999, 7.0, 6.38905621, 15.0]> : tensor<6xf64>
  %result = stablehlo.log_plus_one %operand : tensor<6xf64>
  check.expect_almost_eq_const %result, dense<[0xFFFFFFFFFFFFFFFF, 0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]> : tensor<6xf64>
  func.return
}

// -----

// func.func @log_op_test_c128() {
//   %operand = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
//   %result = stablehlo.log %operand : tensor<complex<f64>>
//   check.expect_almost_eq_const %result, dense<(0.80471895621705025, 1.1071487177940904)> : tensor<complex<f64>>
//   func.return
// }


// %operand: 
// %result = "stablehlo.log_plus_one"(%operand) : (tensor<6xf32>) -> tensor<6xf32>
// %result: 