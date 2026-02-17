// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @fft_op_test_fft_1d_c128() {
  %operand = stablehlo.constant dense<[(1.0, 2.0), (3.0, -4.0)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.fft %operand, type = FFT, length = [2] : (tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(4.0, -2.0), (-2.0, 6.0)]> : tensor<2xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_fft_2d_c128() {
  %operand = stablehlo.constant dense<[[(1.0, 2.0), (3.0, -4.0)], [(-5.0, 6.0), (-7.0, -8.0)]]> : tensor<2x2xcomplex<f64>>
  %result = stablehlo.fft %operand, type = FFT, length = [2, 2] : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[(-8.0, -4.0), (16.0, 0.0)], [(0.0, 20.0), (-4.0, -8.0)]]> : tensor<2x2xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_fft_3d_c128() {
  %operand = stablehlo.constant dense<[[[(1.0, 2.0), (3.0, -4.0)], [(-5.0, 6.0), (-7.0, -8.0)]], [[(-1.0, -2.0), (-3.0, 4.0)], [(5.0, -6.0), (7.0, 8.0)]]]> : tensor<2x2x2xcomplex<f64>>
  %result = stablehlo.fft %operand, type = FFT, length = [2, 2, 2] : (tensor<2x2x2xcomplex<f64>>) -> tensor<2x2x2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]], [[(-16.0, -8.0), (32.0, 0.0)], [(0.0, 40.0), (-8.0, -16.0)]]]> : tensor<2x2x2xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_ifft_1d_c128() {
  %operand = stablehlo.constant dense<[(4.0, -2.0), (-2.0, 6.0)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.fft %operand,
    type = IFFT,
    length = [2]
    : (tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(1.0, 2.0), (3.0, -4.0)]> : tensor<2xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_ifft_2d_c128() {
  %operand = stablehlo.constant dense<[[(-8.0, -4.0), (16.0, 0.0)], [(0.0, 20.0), (-4.0, -8.0)]]> : tensor<2x2xcomplex<f64>>
  %result = stablehlo.fft %operand,
    type = IFFT,
    length = [2, 2]
    : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[(1.0, 2.0), (3.0, -4.0)], [(-5.0, 6.0), (-7.0, -8.0)]]> : tensor<2x2xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_ifft_3d_c128() {
  %operand = stablehlo.constant dense<[[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]], [[(-16.0, -8.0), (32.0, 0.0)], [(0.0, 40.0), (-8.0, -16.0)]]]> : tensor<2x2x2xcomplex<f64>>
  %result = stablehlo.fft %operand,
    type = IFFT,
    length = [2, 2, 2]
    : (tensor<2x2x2xcomplex<f64>>) -> tensor<2x2x2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[[(1.0, 2.0), (3.0, -4.0)], [(-5.0, 6.0), (-7.0, -8.0)]], [[(-1.0, -2.0), (-3.0, 4.0)], [(5.0, -6.0), (7.0, 8.0)]]]> : tensor<2x2x2xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_rfft_1d_f64() {
  %operand = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  %result = stablehlo.fft %operand, type = RFFT, length = [4] : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)]> : tensor<3xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_rfft_2d_f64() {
  %operand = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf64>
  %result = stablehlo.fft %operand, type = RFFT, length = [4, 4] : (tensor<4x4xf64>) -> tensor<4x3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[(136.0, 0.0), (-32.0, 32.0), (-32.0, 0.0)], [(-32.0, -32.0), (-8.0, 8.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0), (-8.0, 0.0)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]]> : tensor<4x3xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_rfft_3d_f64() {
  %operand = stablehlo.constant dense<[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]> : tensor<2x2x4xf64>
  %result = stablehlo.fft %operand, type = RFFT, length = [2, 2, 4] : (tensor<2x2x4xf64>) -> tensor<2x2x3xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[[(136.0, 0.0), (-64.0, 0.0)], [(-32.0, 0.0), (0.0, 0.0)]], [[(-8.0, 8.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]], [[(-8.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]]> : tensor<2x2x3xcomplex<f64>>
  func.return
}

// -----

func.func @fft_op_test_irfft_1d_c128() {
  %operand = stablehlo.constant dense<[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.fft %operand, type = IRFFT, length = [4] : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
  check.expect_almost_eq_const %result, dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  func.return
}

// -----

func.func @fft_op_test_irfft_2d_c128() {
  %operand = stablehlo.constant dense<[[(136.0, 0.0), (-32.0, 32.0), (-32.0, 0.0)], [(-32.0, -32.0), (-8.0, 8.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0), (-8.0, 0.0)], [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]]> : tensor<4x3xcomplex<f64>>
  %result = stablehlo.fft %operand, type = IRFFT, length = [4, 4] : (tensor<4x3xcomplex<f64>>) -> tensor<4x4xf64>
  check.expect_almost_eq_const %result, dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf64>
  func.return
}

// -----

func.func @fft_op_test_irfft_3d_c128() {
  %operand = stablehlo.constant dense<[[[(136.0, 0.0), (-64.0, 0.0)], [(-32.0, 0.0), (0.0, 0.0)]], [[(-8.0, 8.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]], [[(-8.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]]> : tensor<2x2x3xcomplex<f64>>
  %result = stablehlo.fft %operand, type = IRFFT, length = [2, 2, 4] : (tensor<2x2x3xcomplex<f64>>) -> tensor<2x2x4xf64>
  check.expect_almost_eq_const %result, dense<[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]> : tensor<2x2x4xf64>
  func.return
}
