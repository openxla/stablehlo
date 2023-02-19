// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: log_op_test_i64
func.func @log_op_test_i64() -> tensor<2x2xf64> {
  %operand = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %result = stablehlo.log %operand : tensor<2x2xf64>
  func.return %result : tensor<2x2xf64>
  // CHECK-NEXT: tensor<2x2xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.69314718055994529 : f64
  // CHECK-NEXT: 1.0986122886681098 : f64
  // CHECK-NEXT: 1.3862943611198906 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: log_op_test_c128
func.func @log_op_test_c128() -> tensor<complex<f64>> {
  %operand = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %result = stablehlo.log %operand : tensor<complex<f64>>
  func.return %result : tensor<complex<f64>>
  // CHECK-NEXT: tensor<complex<f64>>
  // CHECK-NEXT: [0.80471895621705025 : f64, 1.1071487177940904 : f64]
}
