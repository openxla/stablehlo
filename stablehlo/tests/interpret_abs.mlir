// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: abs_op_test_si64
func.func @abs_op_test_si64() -> tensor<3xi64> {
  %0 = stablehlo.constant dense<[-2, 0, 2]> : tensor<3xi64>
  %1 = stablehlo.abs %0 : tensor<3xi64>
  func.return %1 : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 2 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: abs_op_test_f64
func.func @abs_op_test_f64() -> tensor<3xf64> {
  %0 = stablehlo.constant dense<[23.1, -23.1, -1.1]> : tensor<3xf64>
  %1 = stablehlo.abs %0 : tensor<3xf64>
  func.return %1 : tensor<3xf64>
  // CHECK-NEXT: tensor<3xf64>
  // CHECK-NEXT: 2.310000e+01 : f64
  // CHECK-NEXT: 2.310000e+01 : f64
  // CHECK-NEXT: 1.100000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: abs_op_test_c64
func.func @abs_op_test_c64() -> tensor<f64> {
  %0 = stablehlo.constant dense<(3.0, 4.0)> : tensor<complex<f64>>
  %1 = "stablehlo.abs"(%0) : (tensor<complex<f64>>) -> tensor<f64>
  func.return %1 : tensor<f64>
  // CHECK-NEXT: tensor<f64>
  // CHECK-NEXT: 5.000000e+00 : f64
}
