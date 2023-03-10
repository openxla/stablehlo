// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @exponential_op_test_f64() {
  %operand = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf64>
  %result = stablehlo.exponential %operand : tensor<2x2xf64>
  check.expect_almost_eq_const %result, dense<[[1.000000e+00, 2.7182818284590451], [7.3890560989306504, 20.085536923187668]]> : tensor<2x2xf64>
  func.return
}

// -----

func.func @exponential_op_test_c128() {
  %operand = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %result = stablehlo.exponential %operand : tensor<complex<f64>>
  check.expect_almost_eq_const %result, dense<(-1.1312043837568135, 2.4717266720048188)> : tensor<complex<f64>>
  func.return
}
