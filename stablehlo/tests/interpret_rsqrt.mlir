// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @rsqrt_op_test_f64() {
  %operand = stablehlo.constant dense<[[1.0, 4.0], [9.0, 25.0]]> : tensor<2x2xf64>
  %result = stablehlo.rsqrt %operand : tensor<2x2xf64>
  check.expect_almost_eq_const %result, dense<[[1.000000e+00, 5.000000e-01], [0.33333333333333331, 2.000000e-01]]> : tensor<2x2xf64>
  func.return
}

// -----

func.func @rsqrt_op_test_c128() {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (3.0, 4.0)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.rsqrt %operand : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(0.000000e+00, -1.000000e+00), (4.000000e-01, -2.000000e-01)]> : tensor<2xcomplex<f64>>
  func.return
}
