// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: concatenate
func.func @concatenate() -> tensor<4x2xi64> {
  %input0 = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %input1 = stablehlo.constant dense<[[7, 8]]> : tensor<1x2xi64>
  %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
  func.return %result : tensor<4x2xi64>
  // CHECK-NEXT: tensor<4x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: 7 : i64
  // CHECK-NEXT: 8 : i64
}
