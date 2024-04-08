// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @slice_op() {
  %operand = stablehlo.constant dense<[[0, 0, 1, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 1]]> : tensor<3x6xi64>
  %result = "stablehlo.slice"(%operand) {
    start_indices = array<i64: 0, 2>,
    limit_indices = array<i64: 3, 6>,
    strides = array<i64: 2, 3>
  } : (tensor<3x6xi64>) -> tensor<2x2xi64>
  check.expect_eq_const %result, dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>
  func.return
}
