// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: if
func.func @if() -> (tensor<2xi64>, tensor<2xi64>) {
  %pred = stablehlo.constant dense<false> : tensor<i1>
  %result0, %result1 = "stablehlo.if"(%pred) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    "stablehlo.return"(%0, %0) : (tensor<2xi64>, tensor<2xi64>) -> ()
  }, {
    %false_branch_pred = stablehlo.constant dense<true> : tensor<i1>
    %false_branch_result = "stablehlo.if"(%false_branch_pred) ({
      %1 = stablehlo.constant dense<1> : tensor<2xi64>
      "stablehlo.return"(%1) : (tensor<2xi64>) -> ()
    }, {
      %2 = stablehlo.constant dense<2> : tensor<2xi64>
      "stablehlo.return"(%2) : (tensor<2xi64>) -> ()
    }) : (tensor<i1>) -> tensor<2xi64>
    "stablehlo.return"(%false_branch_result, %false_branch_result) : (tensor<2xi64>, tensor<2xi64>) -> ()
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: 1 : i64
