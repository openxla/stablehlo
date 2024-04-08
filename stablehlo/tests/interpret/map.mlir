// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @map_op_test_si64() {
  %input0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
  %input1 = stablehlo.constant dense<[[4, 5], [6, 7]]> : tensor<2x2xi64>
  %result = "stablehlo.map"(%input0, %input1) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    dimensions = array<i64: 0, 1>
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
  check.expect_eq_const %result, dense<[[0, 5], [12, 21]]> : tensor<2x2xi64>
  func.return
}
