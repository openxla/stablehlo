// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: pad
func.func @pad() -> tensor<7x5xi64> {
  %operand = stablehlo.constant dense<[[0, 0, 0, 0],
                                       [0, 1, 2, 0],
                                       [0, 3, 4, 0],
                                       [0, 5, 6, 0],
                                       [0, 0, 0, 0]]> : tensor<5x4xi64>
  %padding_value = stablehlo.constant dense<-1> : tensor<i64>
  %result = stablehlo.pad %operand, %padding_value, low = [1, -1], high = [1, -1], interior = [0, 1]
    : (tensor<5x4xi64>, tensor<i64>) -> tensor<7x5xi64>
  func.return %result : tensor<7x5xi64>
  // CHECK-NEXT: tensor<7x5xi64>
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: -1 : i64
}
