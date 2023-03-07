// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @floor_op_test_bf16() {
  %0 = stablehlo.constant dense<[0xFF80, -2.5, 0x8001, -0.0, 0.0, 0x0001, 2.5, 0x7F80, 0x7FC0]>  : tensor<9xbf16>
  %1 = stablehlo.floor %0 : tensor<9xbf16>
  check.expect_almost_eq_const %1, dense<[0xFF80, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7F80, 0x7FC0]> : tensor<9xbf16>
  func.return
}

// -----

func.func @floor_op_test_f16() {
  %0 = stablehlo.constant dense<[0xFC00, -2.5, 0x8001, -0.0, 0.0, 0x0001, 2.5, 0x7C00, 0x7E00]>  : tensor<9xf16>
  %1 = stablehlo.floor %0 : tensor<9xf16>
  check.expect_almost_eq_const %1, dense<[0xFC00, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7C00, 0x7E00]> : tensor<9xf16>
  func.return
}

// -----

func.func @floor_op_test_f32() {
  %0 = stablehlo.constant dense<[0xFF800000, -2.5, 0x80000001, -0.0, 0.0, 0x00000001, 2.5, 0x7F800000, 0x7FC00000]>  : tensor<9xf32>
  %1 = stablehlo.floor %0 : tensor<9xf32>
  check.expect_almost_eq_const %1, dense<[0xFF800000, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7F800000, 0x7FC00000]> : tensor<9xf32>
  func.return
}

// -----

func.func @floor_op_test_f64() {
  %0 = stablehlo.constant dense<[0xFFF0000000000000, -2.5, 0x8000000000000001, -0.0, 0.0, 0x0000000000000001, 2.5, 0x7FF0000000000000, 0x7FF8000000000000]>  : tensor<9xf64>
  %1 = stablehlo.floor %0 : tensor<9xf64>
  check.expect_almost_eq_const %1, dense<[0xFFF0000000000000, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF8000000000000]> : tensor<9xf64>
  func.return
}
