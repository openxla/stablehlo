// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @reduce_window() {
  %input = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %init_value = stablehlo.constant dense<0> : tensor<i64>
  %result = "stablehlo.reduce_window"(%input, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    base_dilations = dense<[2, 1]> : tensor<2xi64>,
    padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
    window_dilations = dense<[3, 1]> : tensor<2xi64>,
    window_dimensions = dense<[2, 1]> : tensor<2xi64>,
    window_strides = dense<[4, 1]> : tensor<2xi64>
  } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
  check.expect_eq_const %result, dense<[[0, 0], [3, 4]]> : tensor<2x2xi64>
  func.return
}
