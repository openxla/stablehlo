// RUN: stablehlo-translate --interpret %s

func.func @add_2(%arg0: tensor<i64>) -> tensor<i64> {
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.add %arg0, %0 : tensor<i64>
  func.return %1 : tensor<i64>
}

func.func @main() {
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = func.call @add_2(%0) : (tensor<i64>) -> tensor<i64>
  check.expect_eq_const %1, dense<3> : tensor<i64>
  func.return
}
