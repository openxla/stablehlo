// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @if_ops_true_branch() {
  %pred = stablehlo.constant dense<true> : tensor<i1>
  %result0, %result1 = "stablehlo.if"(%pred) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    stablehlo.return %0, %0 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = stablehlo.constant dense<1> : tensor<2xi64>
    stablehlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  check.expect_eq_const %result0, dense<[0,0]> : tensor<2xi64>
  check.expect_eq_const %result1, dense<[0,0]> : tensor<2xi64>
  func.return
}

// -----

func.func @if_ops_false_branch() {
  %pred = stablehlo.constant dense<false> : tensor<i1>
  %result0, %result1 = "stablehlo.if"(%pred) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    stablehlo.return %0, %0 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = stablehlo.constant dense<1> : tensor<2xi64>
    stablehlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  check.expect_eq_const %result0, dense<[1, 1]> : tensor<2xi64>
  check.expect_eq_const %result1, dense<[1, 1]> : tensor<2xi64>
  func.return
}
