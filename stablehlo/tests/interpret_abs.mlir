// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: abs_op_test_si32
func.func @abs_op_test_si32() -> tensor<3xi32> {
  %0 = stablehlo.constant dense<[-2, 0, 2]> : tensor<3xi32>
  %1 = stablehlo.abs %0 : tensor<3xi32>
  func.return %1 : tensor<3xi32>
  // CHECK-NEXT: tensor<3xi32>
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 2 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: abs_op_test_f32
func.func @abs_op_test_f32() -> tensor<3xf32> {
  %0 = stablehlo.constant dense<[23.1, -23.1, -1.1]> : tensor<3xf32>
  %1 = stablehlo.abs %0 : tensor<3xf32>
  func.return %1 : tensor<3xf32>
  // CHECK-NEXT: tensor<3xf32>
  // CHECK-NEXT: 2.310000e+01 : f32
  // CHECK-NEXT: 2.310000e+01 : f32
  // CHECK-NEXT: 1.100000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: abs_op_test_complexf32
func.func @abs_op_test_complexf32() -> tensor<f32> {
  %0 = stablehlo.constant dense<(3.0, 4.0)> : tensor<complex<f32>>
  %1 = "stablehlo.abs"(%0) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %1 : tensor<f32>
  // CHECK-NEXT: tensor<f32>
  // CHECK-NEXT: 5.000000e+00 : f32
}
