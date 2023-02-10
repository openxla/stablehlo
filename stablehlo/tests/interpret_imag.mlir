// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: imag_op_test_f64
func.func @imag_op_test_f64() -> tensor<11xf64> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  %1 = stablehlo.imag %0 : tensor<11xf64>
  func.return %1 : tensor<11xf64>
  // CHECK-NEXT: tensor<11xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: imag_op_test_c128
func.func @imag_op_test_c128() -> tensor<2xf64> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.imag %0 : (tensor<2xcomplex<f64>>) -> tensor<2xf64>
  func.return %1 : tensor<2xf64>
  // CHECK-NEXT: tensor<2xf64>
  // CHECK-NEXT: 2.500000e+00 : f64
  // CHECK-NEXT: 4.500000e+00 : f64
}
