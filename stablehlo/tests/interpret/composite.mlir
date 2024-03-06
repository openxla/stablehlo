// RUN: stablehlo-translate --interpret %s
// RUN: stablehlo-opt --stablehlo-legalize-composite-to-call  %s | stablehlo-translate --interpret

func.func @add_n.impl(%arg0: tensor<i64>) -> tensor<i64> {
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.add %arg0, %0 : tensor<i64>
  func.return %1 : tensor<i64>
}

func.func @main() {
  %0 = stablehlo.constant dense<1> : tensor<i64>
  %1 = stablehlo.composite "stablehlo.add_n" %0 {
    composite_attributes = { n = 2 : i64 },
    decomposition = @add_n.impl
  } : (tensor<i64>) -> tensor<i64>
  check.expect_eq_const %1, dense<3> : tensor<i64>
  func.return
}
