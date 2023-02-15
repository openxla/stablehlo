// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: if_ops_true_branch
func.func @if_ops_true_branch() -> (tensor<2xi64>, tensor<2xi64>) {
  %pred = stablehlo.constant dense<true> : tensor<i1>
  %result0, %result1 = "stablehlo.if"(%pred) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    stablehlo.return %0, %0 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = stablehlo.constant dense<1> : tensor<2xi64>
    stablehlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 0 : i64
// CHECK-NEXT: 0 : i64
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 0 : i64
// CHECK-NEXT: 0 : i64

// -----

// CHECK-LABEL: Evaluated results of function: if_ops_false_branch
func.func @if_ops_false_branch() -> (tensor<2xi64>, tensor<2xi64>) {
  %pred = stablehlo.constant dense<false> : tensor<i1>
  %result0, %result1 = "stablehlo.if"(%pred) ({
    %0 = stablehlo.constant dense<0> : tensor<2xi64>
    stablehlo.return %0, %0 : tensor<2xi64>, tensor<2xi64>
  }, {
    %1 = stablehlo.constant dense<1> : tensor<2xi64>
    stablehlo.return %1, %1 : tensor<2xi64>, tensor<2xi64>
  }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
  func.return %result0, %result1 : tensor<2xi64>, tensor<2xi64>
}
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: tensor<2xi64>
// CHECK-NEXT: 1 : i64
// CHECK-NEXT: 1 : i64
