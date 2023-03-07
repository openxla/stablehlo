// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @log_op_test_i64() {
  %operand = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %result = stablehlo.log %operand : tensor<2x2xf64>
  check.expect_almost_eq_const %result, dense<[[0.000000e+00, 0.69314718055994529], [1.0986122886681098, 1.3862943611198906]]> : tensor<2x2xf64>
  func.return
}

// -----

func.func @log_op_test_c128() {
  %operand = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %result = stablehlo.log %operand : tensor<complex<f64>>
  check.expect_almost_eq_const %result, dense<(0.80471895621705025, 1.1071487177940904)> : tensor<complex<f64>>
  func.return
}
