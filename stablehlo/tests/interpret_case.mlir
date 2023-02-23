// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: case
func.func @case() -> (tensor<2xi64>, tensor<2xi64>) {
  %index = stablehlo.constant dense<-1> : tensor<i32>
  %result_branch0 = stablehlo.constant dense<0> : tensor<2xi64>
  %result_branch1 = stablehlo.constant dense<1> : tensor<2xi64>
  %result0, %result1 = "stablehlo.case"(%index) ({
    stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
  }, {
    stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: 1 : i64
