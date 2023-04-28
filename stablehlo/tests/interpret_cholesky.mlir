// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @cholesky_op_test_f64() {
  %a = stablehlo.constant dense<[[1.0, 2.0, 3.0],
                                 [2.0, 20.0, 26.0],
                                 [3.0, 26.0, 70.0]]> : tensor<3x3xf64>
  %result = stablehlo.cholesky %a, lower = true : tensor<3x3xf64>
  check.expect_almost_eq_const %result, dense<[[1.0, 0.0, 0.0],
                                               [2.0, 4.0, 0.0],
                                               [3.0, 5.0, 6.0]]> : tensor<3x3xf64>
  func.return
}

// -----

func.func @cholesky_op_test_f64() {
  %a = stablehlo.constant dense<[[1.0, 2.0, 3.0],
                                 [2.0, 20.0, 26.0],
                                 [3.0, 26.0, 70.0]]> : tensor<3x3xf64>
  %result = stablehlo.cholesky %a, lower = false : tensor<3x3xf64>
  check.expect_almost_eq_const %result, dense<[[1.0, 2.0, 3.0],
                                               [0.0, 4.0, 5.0],
                                               [0.0, 0.0, 6.0]]> : tensor<3x3xf64>
  func.return
}

// -----

func.func @cholesky_op_test_f64_batching() {
  %a = stablehlo.constant dense<[[[1.0, 2.0, 3.0],
                                  [2.0, 20.0, 26.0],
                                  [3.0, 26.0, 70.0]],
                                 [[1.0, 2.0, 3.0],
                                  [2.0, 20.0, 26.0],
                                  [3.0, 26.0, 70.0]]]> : tensor<2x3x3xf64>
  %result = stablehlo.cholesky %a, lower = true : tensor<2x3x3xf64>
  check.expect_almost_eq_const %result, dense<[[[1.0, 0.0, 0.0],
                                                [2.0, 4.0, 0.0],
                                                [3.0, 5.0, 6.0]],
                                               [[1.0, 0.0, 0.0],
                                                [2.0, 4.0, 0.0],
                                                [3.0, 5.0, 6.0]]]> : tensor<2x3x3xf64>
  func.return
}

// -----

func.func @cholesky_op_test_c128() {
  %a = stablehlo.constant dense<[[(1.0, 0.0), (0.0, 2.0)],
                                 [(0.0, -2.0), (5.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %result = stablehlo.cholesky %a, lower = true : tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[(1.0, 0.0), (0.0, 0.0)],
                                               [(0.0, -2.0), (1.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  func.return
}

// -----

func.func @cholesky_op_test_c128() {
  %a = stablehlo.constant dense<[[(1.0, 0.0), (0.0, 2.0)],
                                 [(0.0, -2.0), (5.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  %result = stablehlo.cholesky %a, lower = false : tensor<2x2xcomplex<f64>>
  check.expect_almost_eq_const %result, dense<[[(1.0, 0.0), (0.0, 2.0)],
                                               [(0.0, 0.0), (1.0, 0.0)]]> : tensor<2x2xcomplex<f64>>
  func.return
}
