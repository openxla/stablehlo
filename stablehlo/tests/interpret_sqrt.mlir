// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @square_root_op_test_f64() {
  %operand = stablehlo.constant dense<[[0.0, 1.0], [4.0, 9.0]]> : tensor<2x2xf64>
  %result = stablehlo.sqrt %operand : tensor<2x2xf64>
  check.expect_almost_eq_const %result, dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf64>
  func.return
}

// -----

func.func @square_root_op_test_c128() {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (3.0, 4.0)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.sqrt %operand : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(0.000000e+00, 1.000000e+00), (2.000000e+00, 1.000000e+00)]> : tensor<2xcomplex<f64>>
  func.return
}
