// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @imag_op_test_f64() {
  %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %1 = stablehlo.imag %0 : tensor<2xf64>
  check.expect_almost_eq_const %1, dense<[0.0, 0.0]> : tensor<2xf64>
  func.return
}

// -----

func.func @imag_op_test_c128() {
  %0 = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.imag %0 : (tensor<2xcomplex<f64>>) -> tensor<2xf64>
  check.expect_almost_eq_const %1, dense<[2.0, 4.0]> : tensor<2xf64>
  func.return
}
