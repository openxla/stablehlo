// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @select_and_scatter_op_test() {
  %operand = stablehlo.constant dense<[[1, 5],
                                       [2, 5],
                                       [3, 6],
                                       [4, 4]]> : tensor<4x2xi64>
  %source = stablehlo.constant dense<[[5, 6],
                                      [7, 8]]> : tensor<2x2xi64>
  %init_value = stablehlo.constant dense<0> : tensor<i64>
  %result = "stablehlo.select_and_scatter"(%operand, %source, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %0 : tensor<i1>
  }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 2, 1>,
    padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
  } : (tensor<4x2xi64>, tensor<2x2xi64>, tensor<i64>) -> tensor<4x2xi64>
  check.expect_eq_const %result, dense<[[0, 0],
                                        [0, 0],
                                        [5, 14],
                                        [7, 0]]> : tensor<4x2xi64>
  func.return
}
