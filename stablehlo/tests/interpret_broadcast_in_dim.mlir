// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @broadcast_in_dim() {
  %operand = stablehlo.constant dense<[[1], [2], [3]]> : tensor<3x1xi64>
  %result = "stablehlo.broadcast_in_dim"(%operand) {
    broadcast_dimensions = dense<[0, 2]>: tensor<2xi64>
  } : (tensor<3x1xi64>) -> tensor<3x2x2xi64>
  check.expect_eq_const %result, dense<[[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]]> : tensor<3x2x2xi64>
  func.return
}
