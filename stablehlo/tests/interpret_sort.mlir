// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @sort_stable() {
  %input0 = stablehlo.constant dense<[[1, 2, 3], [3, 2, 1]]> : tensor<2x3xi64>
  %input1 = stablehlo.constant dense<[[3, 2, 1], [1, 2, 3]]> : tensor<2x3xi64>
  %result0, %result1 = "stablehlo.sort"(%input0, %input1) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>):
      %predicate = stablehlo.compare GT, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %predicate : tensor<i1>
  }) {
    dimension = 0 : i64,
    is_stable = true
  } : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<2x3xi64>)
  check.expect_eq_const %result0, dense<[[3, 2, 3], [1, 2, 1]]> : tensor<2x3xi64>
  check.expect_eq_const %result1, dense<[[1, 2, 1], [3, 2, 3]]> : tensor<2x3xi64>
  func.return
}
