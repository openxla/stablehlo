// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @square_op_test_f64() {
  %operand = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf64>
  %result = stablehlo.square %operand : tensor<2x2xf64>
  check.expect_almost_eq_const %result, dense<[[0.000000e+00, 1.000000e+00], [4.000000e+00, 9.000000e+00]]> : tensor<2x2xf64>
  func.return
}

// -----

func.func @square_op_test_c128() {
  %operand = stablehlo.constant dense<[(0.0, 1.0), (2.0, 1.0), (1.0e+38, 1.0e+38), (1.0e+155, 2.0e+155), (1.0e+308, 1.0e+308), (1.0e+308, -1.0e+308), (0x7FF0000000000000, 0x7FF0000000000000)]> : tensor<7xcomplex<f64>>
  %result = stablehlo.square %operand : tensor<7xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(-1.000000e+00, 0.000000e+00), (3.000000e+00, 4.000000e+00), (0.000000e+00, 1.9999999999999998E+76), (0xFFF0000000000000, 0x7FF0000000000000), (0.000000e+00,  0x7FF0000000000000), (0.000000e+00,  0xFFF0000000000000), (0xFFF8000000000000,  0x7FF0000000000000)]> : tensor<7xcomplex<f64>>
  func.return
}
