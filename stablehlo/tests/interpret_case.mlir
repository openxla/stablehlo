// RUN: stablehlo-interpreter --interpret -split-input-file %s

func.func @case() {
  %index = stablehlo.constant dense<-1> : tensor<i32>
  %result_branch0 = stablehlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = stablehlo.constant dense<1> : tensor<2xi64>
  %result0, %result1 = "stablehlo.case"(%index) ({
    stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  }, {
    stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  check.eq %result0, dense<[1, 1]> : tensor<2xi64>
  check.eq %result1, dense<[1, 1]> : tensor<2xi64>
  func.return
}
