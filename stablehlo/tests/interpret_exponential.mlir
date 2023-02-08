// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: exponential_op_test_i64
func.func @exponential_op_test_i64() -> tensor<5xf64> {
  %operand = stablehlo.constant dense<[0.0, 1.0, -2.3, 3.1415926535897931, 0x7FF0000000000000]> : tensor<5xf64>
  %result = stablehlo.exponential %operand : tensor<5xf64>
  func.return %result : tensor<5xf64>
  // CHECK-NEXT: tensor<5xf64>
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.7182818284590451 : f64
  // CHECK-NEXT: 0.10025884372280375 : f64
  // CHECK-NEXT: 23.140692632779267 : f64
  // CHECK-NEXT: 0x7FF0000000000000 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: exponential_op_test_c128
func.func @exponential_op_test_c128() -> tensor<4xcomplex<f64>> {
  %operand = stablehlo.constant dense<[(0.0, 0.0), (1.0, 0.0), (1.5, 2.5), (-7.5, -5.5)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.exponential %operand : tensor<4xcomplex<f64>>
  func.return %result : tensor<4xcomplex<f64>>
  // CHECK-NEXT: tensor<4xcomplex<f64>>
  // CHECK-NEXT: [1.000000e+00 : f64, -0.000000e+00 : f64]
  // CHECK-NEXT: [0.54030230586813977 : f64, -0.000000e+00 : f64]
  // CHECK-NEXT: [0.43378099760770306 : f64, -6.0350486377665726 : f64]
  // CHECK-NEXT: [42.41014116569557 : f64, -114.75859669464516 : f64]
}