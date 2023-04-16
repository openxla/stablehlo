// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @add_op_test_f64() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14, 0x7FF0000000000000, 0.0]> : tensor<8xf64>
  %1 = stablehlo.constant dense<[0.0, -0.0, 7.0, 0.75, 0.3, 3.14, 0.0, 0x7FF0000000000000]> : tensor<8xf64>
  %2 = stablehlo.complex %0, %1 : tensor<8xcomplex<f64>>
  check.expect_almost_eq_const %2, dense<[
    (0.0, 0.0),
    (-0.0, -0.0),
    (1.0, 7.0),
    (0.125, 0.75),
    (0.1, 0.3),
    (3.14, 3.14),
    (0x7FF0000000000000, 0.0),
    (0.0, 0x7FF0000000000000)
  ]> : tensor<8xcomplex<f64>>
  func.return
}
