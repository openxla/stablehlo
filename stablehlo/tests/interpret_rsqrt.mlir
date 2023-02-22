// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: rsqrt_op_test_f64
func.func @rsqrt_op_test_f64() -> tensor<2x2xf64> {
  %operand = stablehlo.constant dense<[[1.0, 4.0], [9.0, 25.0]]> : tensor<2x2xf64>
  %result = stablehlo.rsqrt %operand : tensor<2x2xf64>
  func.return %result : tensor<2x2xf64>
  // CHECK-NEXT: tensor<2x2xf64>
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 5.000000e-01 : f64
  // CHECK-NEXT: 0.33333333333333331 : f64
  // CHECK-NEXT: 2.000000e-01 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: rsqrt_op_test_c128
func.func @rsqrt_op_test_c128() -> tensor<2xcomplex<f64>> {
  %operand = stablehlo.constant dense<[(-1.0, 0.0), (3.0, 4.0)]> : tensor<2xcomplex<f64>>
  %result = stablehlo.rsqrt %operand : tensor<2xcomplex<f64>>
  func.return %result : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [0.000000e+00 : f64, -1.000000e+00 : f64]
  // CHECK-NEXT: [4.000000e-01 : f64, -2.000000e-01 : f64]
}
