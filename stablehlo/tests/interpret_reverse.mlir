// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @reverse() {
  %operand = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %result = "stablehlo.reverse"(%operand) {
    dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<3x2xi64>) -> tensor<3x2xi64>
  check.expect_eq_const %result, dense<[[6, 5], [4, 3], [2, 1]]> : tensor<3x2xi64>
  func.return
}
