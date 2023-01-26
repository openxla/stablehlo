// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: dynamic_update_slice
func.func @dynamic_update_slice() -> tensor<4x4xi64> {
  %operand = stablehlo.constant dense<[[1, 1, 0, 0],
                                       [1, 1, 0, 0],
                                       [1, 1, 1, 1],
                                       [1, 1, 1, 1]]> : tensor<4x4xi64>
  %update = stablehlo.constant dense<[[1, 1],
                                      [1, 1]]> : tensor<2x2xi64>
  %start_indices0 = stablehlo.constant dense<-1> : tensor<i64>
  %start_indices1 = stablehlo.constant dense<3> : tensor<i64>
  %result = stablehlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1 :
      (tensor<4x4xi64>, tensor<2x2xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
  func.return %result : tensor<4x4xi64>
  // CHECK-NEXT: tensor<4x4xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
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

// -----

// CHECK-LABEL: Evaluated results of function: dynamic_update_slice_adjusted_start_indices
func.func @dynamic_update_slice_adjusted_start_indices() -> tensor<4x4xi64> {
  %operand = stablehlo.constant dense<[[1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       [1, 2, 2, 2],
                                       [1, 2, 2, 2]]> : tensor<4x4xi64>
  %update = stablehlo.constant dense<[[1, 1, 1],
                                      [1, 1, 1]]> : tensor<2x3xi64>
  %start_indices0 = stablehlo.constant dense<4> : tensor<i64>
  %start_indices1 = stablehlo.constant dense<4> : tensor<i64>
  %result = stablehlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1 :
      (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
  func.return %result : tensor<4x4xi64>
  // CHECK-NEXT: tensor<4x4xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
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

// -----

// CHECK-LABEL: Evaluated results of function: dynamic_update_slice_empty
func.func @dynamic_update_slice_empty() -> tensor<2x2xi64> {
  %operand = stablehlo.constant dense<[[1, 2],
                                       [3, 4]]> : tensor<2x2xi64>
  %update = stablehlo.constant dense<> : tensor<0x0xi64>
  %start_indices0 = stablehlo.constant dense<0> : tensor<i64>
  %start_indices1 = stablehlo.constant dense<0> : tensor<i64>
  %result = stablehlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1 :
      (tensor<2x2xi64>, tensor<0x0xi64>, tensor<i64>, tensor<i64>) -> tensor<2x2xi64>
  func.return %result : tensor<2x2xi64>
  // CHECK-NEXT: tensor<2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 4 : i64
}
