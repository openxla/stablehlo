// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @reduce() {
  %input = stablehlo.constant dense<[[0, 1, 2, 3, 4, 5]]> : tensor<1x6xi64>
  %init_value = stablehlo.constant dense<0> : tensor<i64>
  %result = "stablehlo.reduce"(%input, %init_value) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %0 : tensor<i64>
  }) {
    dimensions = array<i64: 1>
  } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
  check.expect_eq_const %result, dense<[15]> : tensor<1xi64>
  func.return
}
