// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: remainder_op_test_si64
func.func @remainder_op_test_si64() -> tensor<4xi64> {
  %lhs = stablehlo.constant dense<[17, -17, 17, -17]> : tensor<4xi64>
  %rhs = stablehlo.constant dense<[3, 3, -3, -3]> : tensor<4xi64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
  func.return %result : tensor<4xi64>
  // CHECK-NEXT: tensor<4xi64>
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: -2 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: -2 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: remainder_op_test_ui64
func.func @remainder_op_test_ui64() -> tensor<4xui64> {
  %lhs = stablehlo.constant dense<[17, 18, 19, 20]> : tensor<4xui64>
  %rhs = stablehlo.constant dense<[3, 4, 5, 7]> : tensor<4xui64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<4xui64>
  func.return %result : tensor<4xui64>
  // CHECK-NEXT: tensor<4xui64>
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 6 : ui64
}

// -----

// CHECK-LABEL: Evaluated results of function: remainder_op_test_f64
func.func @remainder_op_test_f64() -> tensor<5xf64> {
  %lhs = stablehlo.constant dense<[1.0, 17.1, -17.1, 17.1, -17.1]> : tensor<5xf64>
  %rhs = stablehlo.constant dense<[0.0, 3.0, 3.0, -3.0, -3.0]> : tensor<5xf64>
  %result = stablehlo.remainder %lhs, %rhs : tensor<5xf64>
  func.return %result : tensor<5xf64>
  // CHECK-NEXT: 0x7FF8000000000000 : f64
  // CHECK-NEXT: tensor<5xf64>
  // CHECK-NEXT: 2.1000000000000014 : f64
  // CHECK-NEXT: -2.1000000000000014 : f64
  // CHECK-NEXT: 2.1000000000000014 : f64
  // CHECK-NEXT: -2.1000000000000014 : f64
}
