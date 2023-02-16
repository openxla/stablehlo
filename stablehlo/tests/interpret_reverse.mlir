// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: reverse
func.func @reverse() -> tensor<3x2xi64> {
  %operand = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %result = "stablehlo.reverse"(%operand) {
    dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<3x2xi64>) -> tensor<3x2xi64>
  func.return %result : tensor<3x2xi64>
  // CHECK-NEXT: tensor<3x2xi64>
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 1 : i64
}
