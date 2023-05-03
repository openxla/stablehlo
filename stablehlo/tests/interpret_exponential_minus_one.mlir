// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @exponential_minus_one_op_test_f64() {
  %operand = stablehlo.constant dense<[0.0, 1.0]> : tensor<2xf64>
  %result = stablehlo.exponential_minus_one %operand : tensor<2xf64>
  check.expect_almost_eq_const %result, dense<[0.0, 1.7182818284590451]> : tensor<2xf64>
  func.return
}

// -----

func.func @exponential_minus_one_op_test_c128() {
  %operand = stablehlo.constant dense<[(1.0, 2.0), (2.0, 1.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.exponential_minus_one %operand : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[
	(-2.13120438375681363, 2.47172667200481892),
	(2.99232404844127142, 6.21767631236796820),
	(0.0, 0.0)
  ]> : tensor<3xcomplex<f64>>
  func.return
}
