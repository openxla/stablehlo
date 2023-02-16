// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: dynamic_slice
func.func @dynamic_slice() -> tensor<3x3xi64> {
  %operand = stablehlo.constant dense<[[1, 1, 1],
                                       [1, 1, 1],
                                       [1, 1, 1]]> : tensor<3x3xi64>
  %start_indices0 = stablehlo.constant dense<3> : tensor<i64>
  %start_indices1 = stablehlo.constant dense<3> : tensor<i64>
  %result = "stablehlo.dynamic_slice"(%operand, %start_indices0, %start_indices1) {
    slice_sizes = dense<[3, 3]> : tensor<2xi64>
  } : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
  func.return %result : tensor<3x3xi64>
  // CHECK-NEXT: tensor<3x3xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
}
