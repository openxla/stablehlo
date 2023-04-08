// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @cosine_op_test_bf16() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xbf16>
  %1 = stablehlo.cosine %0 : tensor<11xbf16>
  check.expect_almost_eq_const %1, dense<[1.000000e+00, 1.000000e+00, 5.390630e-01, 9.921870e-01, 9.960930e-01, -1.000000e+00, 0xFFC0, 0xFFC0, 0x7FFF, 1.000000e+00, 1.000000e+00]> : tensor<11xbf16>
  func.return
}

// -----

func.func @cosine_op_test_f16() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7C00, 0xFC00, 0x7FFF, 0x0001, 0x8001]> : tensor<11xf16>
  %1 = stablehlo.cosine %0 : tensor<11xf16>
  check.expect_almost_eq_const %1, dense<[1.000000e+00, 1.000000e+00, 5.405270e-01, 9.921870e-01, 9.951170e-01, -1.000000e+00, 0xFE00, 0xFE00, 0x7FFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf16>
  func.return
}

// -----

func.func @cosine_op_test_f32() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 0x00000001, 0x80000001]> : tensor<11xf32>
  %1 = stablehlo.cosine %0 : tensor<11xf32>
  check.expect_almost_eq_const %1, dense<[1.000000e+00, 1.000000e+00, 0.540302277, 0.992197692, 0.995004177, -1.000000e+00, 0xFFC00000, 0xFFC00000, 0x7FFFFFFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf32>
  func.return
}

// -----

func.func @cosine_op_test_f64() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  %1 = stablehlo.cosine %0 : tensor<11xf64>
  check.expect_almost_eq_const %1, dense<[1.000000e+00, 1.000000e+00, 0.54030230586813977, 0.992197667229329, 0.99500416527802582, -1.000000e+00, 0xFFF8000000000000, 0xFFF8000000000000, 0x7FFFFFFFFFFFFFFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf64>
  func.return
}

// -----

func.func @cosine_op_test_c64() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  %1 = stablehlo.cosine %0 : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %1, dense<[(4.337810e-01, -6.03504848), (-42.1537743, 15.7863016)]> : tensor<2xcomplex<f32>>
  func.return
}

// -----

func.func @cosine_op_test_c128() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.cosine %0 : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %1, dense<[(0.43378099760770306, -6.0350486377665726), (-42.153773835602316, 15.786301507647636)]> : tensor<2xcomplex<f64>>
  func.return
}
