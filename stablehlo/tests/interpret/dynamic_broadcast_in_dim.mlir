// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @broadcast_in_dim() {
  %operand = stablehlo.constant dense<[[1], [2], [3]]> : tensor<3x1xi64>
  %output_dimensions = stablehlo.constant dense<[3, 2, 2]> : tensor<3xi64>
  %result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
  broadcast_dimensions = array<i64: 0, 2>,
  known_expanding_dimensions = array<i64: 1>,
  known_nonexpanding_dimensions = array<i64: 0>
} : (tensor<3x1xi64>, tensor<3xi64>) -> tensor<3x2x2xi64>
  check.expect_eq_const %result, dense<[
                                        [
                                          [1, 1],
                                          [1, 1]
                                        ],
                                        [
                                          [2, 2],
                                          [2, 2]
                                        ],
                                        [
                                          [3, 3],
                                          [3, 3]
                                        ]
                                        ]> : tensor<3x2x2xi64>
  func.return
}
