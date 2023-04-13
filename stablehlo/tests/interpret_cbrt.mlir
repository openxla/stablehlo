// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @cbrt_op_test_f64() {
  %operand = stablehlo.constant dense<[0.0, 1.0, 8.0, 27.0]> : tensor<4xf64>
  %result = stablehlo.cbrt %operand : tensor<4xf64>
  check.expect_almost_eq_const %result, dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf64>
  func.return
}

// -----

func.func @cbrt_op_test_c128() {
  %operand = stablehlo.constant dense<(3.0, 4.0)> : tensor<complex<f64>>
  %result = stablehlo.cbrt %operand : tensor<complex<f64>>
  check.expect_almost_eq_const %result, dense<(1.6289371459221759, 0.5201745023045458)> : tensor<complex<f64>>
  func.return
}
