// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: broadcast_in_dim
func.func @broadcast_in_dim() -> tensor<3x2x2xi64> {
  %operand = stablehlo.constant dense<[[1], [2], [3]]> : tensor<3x1xi64>
  %result = "stablehlo.broadcast_in_dim"(%operand) {
    broadcast_dimensions = dense<[0, 2]>: tensor<2xi64>
  } : (tensor<3x1xi64>) -> tensor<3x2x2xi64>
  func.return %result : tensor<3x2x2xi64>
  // CHECK-NEXT: tensor<3x2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 3 : i64
}
