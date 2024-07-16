// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @tan_op_test_f64() {
  %0 = stablehlo.constant dense<[0.0, 1.57079632, 3.14159265, 4.71238898]> : tensor<4xf64>
  %1 = stablehlo.tan %0 : tensor<4xf64>
  check.expect_almost_eq_const %1, dense<[0.000000e+00, 147169271.76124874, -3.5897930298416118E-9,  2599497068.2695704]> : tensor<4xf64>
  func.return
}
