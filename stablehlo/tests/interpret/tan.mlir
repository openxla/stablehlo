// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @tan_op_test_bf16() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xbf16>
  %1 = stablehlo.tan %0 : tensor<11xbf16>
  check.expect_almost_eq_const %1, dense<[0.000000e+00, -0.000000e+00, 1.554690e+00, 1.259770e-01, 1.005860e-01, -9.689330e-04, 0xFFC0, 0xFFC0, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>
  func.return
}

// -----

func.func @tan_op_test_f16() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xf16>
  %1 = stablehlo.tan %0 : tensor<11xf16>
  check.expect_almost_eq_const %1, dense<[0.000000e+00, -0.000000e+00, 1.557620e+00, 1.256100e-01, 1.002810e-01, -9.675020e-04, 0x7F80, 0xFF80, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>
  func.return
}

// -----

func.func @tan_op_test_f32() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 0x00000001, 0x80000001]> : tensor<11xf32>
  %1 = stablehlo.tan %0 : tensor<11xf32>
  check.expect_almost_eq_const %1, dense<[0.000000e+00, -0.000000e+00,  1.55740774, 0.12565513, 0.100334674, 8.74227765E-8, 0xFFC00000, 0xFFC00000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>
  func.return
}

// -----

func.func @tan_op_test_f64() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  %1 = stablehlo.tan %0 : tensor<11xf64>
  check.expect_almost_eq_const %1, dense<[0.000000e+00, -0.000000e+00, 1.5574077246549023,  0.12565513657513097, 0.10033467208545055, -1.2246467991473532E-16, 0xFFF8000000000000, 0xFFF8000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>
  func.return
}

// -----

func.func @tan_op_test_c32() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  %1 = stablehlo.tan %0 : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %1, dense<[(0.00192734355, 1.01342881), (1.621270e-04, 0.999813914)]> : tensor<2xcomplex<f32>>
  func.return
}

// -----

func.func @tan_op_test_c64() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.tan %0 : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %1, dense<[(0.0019273435237456358, 1.0134287782038933), (1.6212700415590609E-4, 0.99981392630805066)]> : tensor<2xcomplex<f64>>
  func.return
}

// ToDo: do we need tests for non IEEE-754 floats?