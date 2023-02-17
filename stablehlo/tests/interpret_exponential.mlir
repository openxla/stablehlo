// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: exponential_op_test_f64
func.func @exponential_op_test_f64() -> tensor<2x2xf64> {
  %operand = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf64>
  %result = stablehlo.exponential %operand : tensor<2x2xf64>
  func.return %result : tensor<2x2xf64>
  // CHECK-NEXT: tensor<2x2xf64>
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.7182818284590451 : f64
  // CHECK-NEXT: 7.3890560989306504 : f64
  // CHECK-NEXT: 20.085536923187668 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: exponential_op_test_c128
func.func @exponential_op_test_c128() -> tensor<complex<f64>> {
  %operand = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %result = stablehlo.exponential %operand : tensor<complex<f64>>
  func.return %result : tensor<complex<f64>>
  // CHECK-NEXT: tensor<complex<f64>>
  // CHECK-NEXT: [-1.1312043837568135 : f64, 2.4717266720048188 : f64]
}
