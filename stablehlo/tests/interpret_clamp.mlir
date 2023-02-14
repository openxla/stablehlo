// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si64
func.func @clamp_op_test_si64() -> tensor<3xi64> {
  %min = stablehlo.constant dense<[1, 5, -5]> : tensor<3xi64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<[3, 7, -3]> : tensor<3xi64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 5 : i64
  // CHECK-NEXT: -3 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si64_min_scalar
func.func @clamp_op_test_si64_min_scalar() -> tensor<3xi64> {
  %min = stablehlo.constant dense<[0, 0, -2]> : tensor<3xi64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<1> : tensor<i64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<3xi64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: -1 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si64_max_scalar
func.func @clamp_op_test_si64_max_scalar() -> tensor<3xi64> {
  %min = stablehlo.constant dense<0> : tensor<i64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<[1, 1, 4]> : tensor<3xi64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<i64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: clamp_op_test_si64_min_max_both_scalar
func.func @clamp_op_test_si64_min_max_both_scalar() -> tensor<3xi64> {
  %min = stablehlo.constant dense<0> : tensor<i64>
  %operand = stablehlo.constant dense<[2, 3, -1]> : tensor<3xi64>
  %max = stablehlo.constant dense<1> : tensor<i64>
  %result = stablehlo.clamp %min, %operand, %max : (tensor<i64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 0 : i64
}
