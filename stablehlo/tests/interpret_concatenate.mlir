// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: concatenate
func.func @concatenate() -> tensor<2x3x2xi64> {
  %input0 = stablehlo.constant dense<[[[1, 2]], [[7, 8]]]> : tensor<2x1x2xi64>
  %input1 = stablehlo.constant dense<> : tensor<2x0x2xi64>
  %input2 = stablehlo.constant dense<[[[3, 4], [5, 6]], [[9, 10], [11, 12]]]> : tensor<2x2x2xi64>
  %result = stablehlo.concatenate %input0, %input1, %input2, dim = 1 : (tensor<2x1x2xi64>, tensor<2x0x2xi64>, tensor<2x2x2xi64>) -> tensor<2x3x2xi64>
  func.return %result : tensor<2x3x2xi64>
  // CHECK-NEXT: tensor<2x3x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: 7 : i64
  // CHECK-NEXT: 8 : i64
  // CHECK-NEXT: 9 : i64
  // CHECK-NEXT: 10 : i64
  // CHECK-NEXT: 11 : i64
  // CHECK-NEXT: 12 : i64
}
