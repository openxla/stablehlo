// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: pad
func.func @pad() -> tensor<5x9xi64> {
  %operand = stablehlo.constant dense<[[1, 2, 3],
                                       [4, 5, 6]]> : tensor<2x3xi64>
  %padding_value = stablehlo.constant dense<0> : tensor<i64>
  %result = stablehlo.pad %operand, %padding_value, low = [0, 1], high = [2, 1], interior = [1, 2]
    : (tensor<2x3xi64>, tensor<i64>) -> tensor<5x9xi64>
  func.return %result : tensor<5x9xi64>
  // CHECK-NEXT: tensor<5x9xi64>
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: pad_negative_padding
func.func @pad_negative_padding() -> tensor<2x3xi64> {
  %operand = stablehlo.constant dense<[[0, 1, 2, 3, 0],
                                       [0, 4, 5, 6, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]> : tensor<4x5xi64>
  %padding_value = stablehlo.constant dense<0> : tensor<i64>
  %result = stablehlo.pad %operand, %padding_value, low = [0, -1], high = [-2, -1], interior = [0, 0]
    : (tensor<4x5xi64>, tensor<i64>) -> tensor<2x3xi64>
  func.return %result : tensor<2x3xi64>
  // CHECK-NEXT: tensor<2x3xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 6 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: pad_negative_padding_with_interior_dim_0
func.func @pad_negative_padding_with_interior_dim_0() -> tensor<3x3xi64> {
  %operand = stablehlo.constant dense<[[0, 1, 2, 3, 0],
                                       [0, 4, 5, 6, 0],
                                       [0, 0, 0, 0, 0]]> : tensor<3x5xi64>
  %padding_value = stablehlo.constant dense<0> : tensor<i64>
  %result = stablehlo.pad %operand, %padding_value, low = [0, -1], high = [-2, -1], interior = [1, 0]
    : (tensor<3x5xi64>, tensor<i64>) -> tensor<3x3xi64>
  func.return %result : tensor<3x3xi64>
  // CHECK-NEXT: tensor<3x3xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 6 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: pad_negative_padding_with_interior_dim_1
func.func @pad_negative_padding_with_interior_dim_1() -> tensor<2x7xi64> {
  %operand = stablehlo.constant dense<[[0, 1, 2, 3, 0],
                                       [0, 4, 5, 6, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]> : tensor<4x5xi64>
  %padding_value = stablehlo.constant dense<0> : tensor<i64>
  %result = stablehlo.pad %operand, %padding_value, low = [0, -1], high = [-2, -1], interior = [0, 1]
    : (tensor<4x5xi64>, tensor<i64>) -> tensor<2x7xi64>
  func.return %result : tensor<2x7xi64>
  // CHECK-NEXT: tensor<2x7xi64>
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 4 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 6 : i64
  // CHECK-NEXT: 0 : i64
}
