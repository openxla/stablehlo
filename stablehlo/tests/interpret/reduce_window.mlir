// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @reduce_window() {
  %input = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %init_value = stablehlo.constant dense<0> : tensor<i64>
  %result = "stablehlo.reduce_window"(%input, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    base_dilations = array<i64: 2, 1>,
    padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
    window_dilations = array<i64: 3, 1>,
    window_dimensions = array<i64: 2, 1>,
    window_strides = array<i64: 4, 1>
  } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
  check.expect_eq_const %result, dense<[[0, 0], [3, 4]]> : tensor<2x2xi64>
  func.return
}

// -----

func.func @reduce_window_issue_1662() {
  %input = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %init_value = stablehlo.constant dense<0> : tensor<i64>
  %result = "stablehlo.reduce_window"(%input, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    base_dilations = array<i64: 2, 1>,
    padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
    window_dilations = array<i64: 3, 1>,
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 4, 1>
  } : (tensor<3x2xi64>, tensor<i64>) -> tensor<1x2xi64>
  check.expect_eq_const %result, dense<[[5, 6]]> : tensor<1x2xi64>
  func.return
}
