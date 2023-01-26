// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: slice_op
func.func @slice_op() -> tensor<2x2xi64> {
  %operand = stablehlo.constant dense<[[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]> : tensor<3x4xi64>
  %result = "stablehlo.slice"(%operand) {
    start_indices = dense<[1, 2]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi64>) -> tensor<2x2xi64>
  func.return %result : tensor<2x2xi64>
  // CHECK-NEXT: tensor<2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: slice_op_1d
func.func @slice_op_1d() -> tensor<2xi64> {
  %operand = stablehlo.constant dense<[0, 1, 2, 3, 4]> : tensor<5xi64>
  %result = "stablehlo.slice"(%operand) {
    start_indices = dense<2> : tensor<1xi64>,
    limit_indices = dense<4> : tensor<1xi64>,
    strides = dense<1> : tensor<1xi64>
  } : (tensor<5xi64>) -> tensor<2xi64>
  func.return %result : tensor<2xi64>
  // CHECK-NEXT: tensor<2xi64>
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: slice_op_2d
func.func @slice_op_2d() -> tensor<2x2xi64> {
  %operand = stablehlo.constant dense<[[0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]> : tensor<3x6xi64>
  %result = "stablehlo.slice"(%operand) {
    start_indices = dense<[0, 2]> : tensor<2xi64>,
    limit_indices = dense<[3, 6]> : tensor<2xi64>,
    strides = dense<[2, 3]> : tensor<2xi64>
  } : (tensor<3x6xi64>) -> tensor<2x2xi64>
  func.return %result : tensor<2x2xi64>
  // CHECK-NEXT: tensor<2x2xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: slice_op_empty
func.func @slice_op_empty() -> tensor<0xi64> {
  %operand = stablehlo.constant dense<[]> : tensor<0xi64>
  %result = "stablehlo.slice"(%operand) {
    start_indices = dense<0> : tensor<1xi64>,
    limit_indices = dense<0> : tensor<1xi64>,
    strides = dense<1> : tensor<1xi64>
  } : (tensor<0xi64>) -> tensor<0xi64>
  func.return %result : tensor<0xi64>
  // CHECK-NEXT: tensor<0xi64>
}
