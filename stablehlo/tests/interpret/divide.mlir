// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @divide_op_test_si64() {
  %lhs = stablehlo.constant dense<[17, -17, 17, -17]> : tensor<4xi64>
  %rhs = stablehlo.constant dense<[3, 3, -3, -3]> : tensor<4xi64>
  %result = stablehlo.divide %lhs, %rhs : tensor<4xi64>
  check.expect_eq_const %result, dense<[5, -5, -5, 5]> : tensor<4xi64>
  func.return
}

// -----

func.func @divide_op_test_ui64() {
  %lhs = stablehlo.constant dense<[17, 18, 19, 20]> : tensor<4xui64>
  %rhs = stablehlo.constant dense<[3, 4, 5, 7]> : tensor<4xui64>
  %result = stablehlo.divide %lhs, %rhs : tensor<4xui64>
  check.expect_eq_const %result, dense<[5, 4, 3, 2]> : tensor<4xui64>
  func.return
}

// -----

func.func @divide_op_test_f64() {
  %lhs = stablehlo.constant dense<[17.1, -17.1, 17.1, -17.1]> : tensor<4xf64>
  %rhs = stablehlo.constant dense<[3.0, 3.0, -3.0, -3.0]> : tensor<4xf64>
  %result = stablehlo.divide %lhs, %rhs : tensor<4xf64>
  check.expect_almost_eq_const %result, dense<[5.700000e+00, -5.700000e+00, -5.700000e+00, 5.700000e+00]> : tensor<4xf64>
  func.return
}

// -----

func.func @divide_op_test_c128() {
  %lhs = stablehlo.constant dense<[(1.5, 2.5), (7.5, 5.5)]> : tensor<2xcomplex<f64>>
  %rhs = stablehlo.constant dense<[(2.5, 1.5), (5.5, 7.5)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.divide %lhs, %rhs : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(0.88235294117647056, 0.4705882352941177), (0.95375722543352603, -0.30057803468208094)]> : tensor<2xcomplex<f64>>
  func.return
}
