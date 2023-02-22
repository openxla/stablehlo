// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: imag_op_test_f64
func.func @imag_op_test_f64() -> tensor<2xf64> {
  %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %1 = stablehlo.imag %0 : tensor<2xf64>
  func.return %1 : tensor<2xf64>
  // CHECK-NEXT: tensor<2xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: imag_op_test_c128
func.func @imag_op_test_c128() -> tensor<2xf64> {
  %0 = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.imag %0 : (tensor<2xcomplex<f64>>) -> tensor<2xf64>
  func.return %1 : tensor<2xf64>
  // CHECK-NEXT: tensor<2xf64>
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 4.000000e+00 : f64
}
