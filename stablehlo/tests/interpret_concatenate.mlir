// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @concatenate() {
  %input0 = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %input1 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
  %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
  check.expect_eq_const %result, dense<[[1, 2], [3, 4] , [5, 6], [7, 8]]> : tensor<4x2xi64>
  func.return
}
