// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @power_op_test_si64() {
  %lhs = stablehlo.constant dense<[-1, -1, -3, -3, 0]> : tensor<5xi64>
  %rhs = stablehlo.constant dense<[1, 0, -3, 3, 2]> : tensor<5xi64>
  %result = stablehlo.power %lhs, %rhs : tensor<5xi64>
  check.expect_eq_const %result, dense<[-1, 1, 0, -27, 0]> : tensor<5xi64>
  func.return
}

// -----

func.func @power_op_test_ui64() {
  %lhs = stablehlo.constant dense<[0, 0, 1, 1, 5]> : tensor<5xui64>
  %rhs = stablehlo.constant dense<[0, 1, 0, 2, 5]> : tensor<5xui64>
  %result = stablehlo.power %lhs, %rhs : tensor<5xui64>
  check.expect_eq_const %result, dense<[1, 0, 1, 1, 3125]> : tensor<5xui64>
  func.return
}

// -----

func.func @power_op_test_f64() {
  %lhs = stablehlo.constant dense<[-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]> : tensor<6xf64>
  %rhs = stablehlo.constant dense<[2.0, 2.0, 1.1, 2.0, -1.0, 10.0]> : tensor<6xf64>
  %result = stablehlo.power %lhs, %rhs : tensor<6xf64>
  check.expect_almost_eq_const %result, dense<[4.000000e+00, 0.000000e+00, 0xFFF8000000000000,
                                               2.500000e+01, 0.33333333333333331, 1.000000e+40]> : tensor<6xf64>
  func.return
}

// -----

func.func @power_op_test_c128() {
  %lhs = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5)]> : tensor<2xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(2.5, 1.5), (5.5, 7.5)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.power %lhs, %rhs : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(-1.5679313814305016, -2.6674775446623613),
                                               (392.89270835580857, 1801.8249193362644)]> : tensor<2xcomplex<f64>>
  func.return
}
