// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: exponential_op_test_i64
func.func @exponential_op_test_i64() -> tensor<4xf64> {
  %0 = stablehlo.constant dense<[0.0, 1.0, 3.1415926535897931, 0x7FF0000000000000]> : tensor<4xf64>
  %1 = stablehlo.exponential %0 : tensor<4xf64>
  func.return %1 : tensor<4xf64>
  // CHECK-NEXT: tensor<4xf64>
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.7182818284590451 : f64
  // CHECK-NEXT: 23.140692632779267 : f64
  // CHECK-NEXT: 0x7FF0000000000000 : f64
}
