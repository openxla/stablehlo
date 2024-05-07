// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @dynamic_reshape_op_test_si64() {
  %0 = stablehlo.constant dense<[[1,2,3,4,5,6]]> : tensor<1x6xi64>
  %shape = stablehlo.constant dense<[6]> : tensor<1xi64>
  %1 = stablehlo.dynamic_reshape %0, %shape : (tensor<1x6xi64>, tensor<1xi64>) -> tensor<6xi64>
  check.expect_eq_const %1, dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>
  func.return
}