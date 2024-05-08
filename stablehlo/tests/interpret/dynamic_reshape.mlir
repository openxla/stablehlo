// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dynamic_reshape_op_test_si64() {
  %operand = stablehlo.constant dense<[[1, 2, 3, 4, 5, 6]]> : tensor<1x6xi64>
  %output_shape = stablehlo.constant dense<[6]> : tensor<1xi64>
  %result = stablehlo.dynamic_reshape %operand, %output_shape : (tensor<1x6xi64>, tensor<1xi64>) -> tensor<6xi64>
  check.expect_eq_const %result, dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>
  func.return
}
