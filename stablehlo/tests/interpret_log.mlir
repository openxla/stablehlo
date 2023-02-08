// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: log_op_test_i64
func.func @log_op_test_i64() -> tensor<5xf64> {
  %operand = stablehlo.constant dense<[0.0, 1.0, -1.0, 3.1415926535897931, 0x7FF0000000000000]> : tensor<5xf64>
  %result = stablehlo.log %operand : tensor<5xf64>
  func.return %result : tensor<5xf64>
  // CHECK-NEXT: tensor<5xf64>
  // CHECK-NEXT: 0xFFF0000000000000 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0xFFF8000000000000 : f64
  // CHECK-NEXT: 1.1447298858494002 : f64
  // CHECK-NEXT: 0x7FF0000000000000 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: log_op_test_c128
func.func @log_op_test_c128() -> tensor<4xcomplex<f64>> {
  %operand = stablehlo.constant dense<[(0.0, 0.0), (1.0, 0.0), (1.5, 2.5), (-7.5, -5.5)]> : tensor<4xcomplex<f64>>
  %result = stablehlo.log %operand : tensor<4xcomplex<f64>>
  func.return %result : tensor<4xcomplex<f64>>
  // CHECK-NEXT: tensor<4xcomplex<f64>>
  // CHECK-NEXT: [0xFFF0000000000000 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.0700330817481354 : f64, 1.0303768265243125 : f64]
  // CHECK-NEXT: [2.2300722069689169 : f64, -2.5088438185876103 : f64]
}