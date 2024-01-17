// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @atan2_op_test_f64() {
  %0 = stablehlo.constant dense<[0.0, 1.0, -1.0]> : tensor<3xf64>
  %1 = stablehlo.constant dense<[0.0, 0.0, 0.0]> : tensor<3xf64>
  %2 = stablehlo.atan2 %0, %1 : tensor<3xf64>
  check.expect_almost_eq_const %2, dense<[0.0, 1.5707963267948966, -1.5707963267948966]> : tensor<3xf64>
  func.return
}

// -----

func.func @atan2_op_test_c128() {
  %0 = stablehlo.constant dense<[(1.0, 0.0), (-1.0, 0.0)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0)]> : tensor<2xcomplex<f64>>
  %2 = stablehlo.atan2 %0, %1 : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %2, dense<[(1.5707963267948966, -0.0),
                                          (-1.5707963267948966, 0.0)]> : tensor<2xcomplex<f64>>
  func.return
}
