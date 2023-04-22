// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @log_plus_one_op_test_f64() {
  %operand = stablehlo.constant dense<[0.0, -0.999, 7.0, 6.38905621, 15.0]> : tensor<5xf64>
  %result = stablehlo.log_plus_one %operand : tensor<5xf64>
  check.expect_almost_eq_const %result, dense<[0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]> : tensor<5xf64>
  func.return
}

// -----

func.func @log_plus_one_op_test_c128() {
  %operand = stablehlo.constant dense<[(1.0, 2.0), (2.0, 1.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.log_plus_one %operand : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[
	(1.03972077083991, 0.78539816339744),
	(1.15129254649702, 0.32175055439664),
	(0.0, 0.0)
  ]> : tensor<3xcomplex<f64>>
  func.return
}
