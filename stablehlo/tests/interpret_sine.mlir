// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: sine_op_test_bf16
func.func @sine_op_test_bf16() -> tensor<bf16> {
  %0 = stablehlo.constant dense<0.0> : tensor<bf16>
  %1 = stablehlo.sine %0 : tensor<bf16>
  func.return %1 : tensor<bf16>
  // CHECK-NEXT: tensor<bf16>
  // CHECK-NEXT: 0.000000e+00 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_f16
func.func @sine_op_test_f16() -> tensor<f16> {
  %0 = stablehlo.constant dense<0.0> : tensor<f16>
  %1 = stablehlo.sine %0 : tensor<f16>
  func.return %1 : tensor<f16>
  // CHECK-NEXT: tensor<f16>
  // CHECK-NEXT: 0.000000e+00 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_f32_1
func.func @sine_op_test_f32_1() -> tensor<f32> {
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.sine %0 : tensor<f32>
  func.return %1 : tensor<f32>
  // CHECK-NEXT: tensor<f32>
  // CHECK-NEXT: 0.000000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_f32_2
func.func @sine_op_test_f32_2() -> tensor<f32> {
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.sine %0 : tensor<f32>
  func.return %1 : tensor<f32>
  // CHECK-NEXT: tensor<f32>
  // CHECK-NEXT: 0.000000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_f32_3
func.func @sine_op_test_f32_3() -> tensor<4xf32> {
  %0 = stablehlo.constant dense<[0.0, 1.57079632, 3.14159265, 4.71238898]> : tensor<4xf32>
  %1 = stablehlo.sine %0 : tensor<4xf32>
  func.return %1 : tensor<4xf32>
  // CHECK-NEXT: tensor<4xf32>
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: -8.74227765E-8 : f32
  // CHECK-NEXT: -1.000000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_c64
func.func @sine_op_test_c64() -> tensor<complex<f32>> {
  %0 = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f32>>
  %1 = stablehlo.sine %0 : tensor<complex<f32>>
  func.return %1 : tensor<complex<f32>>
  // CHECK-NEXT: tensor<complex<f32>>
  // CHECK-NEXT: [3.16577864 : f32, 1.95960093 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_c128
func.func @sine_op_test_c128() -> tensor<complex<f64>> {
  %0 = stablehlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %1 = stablehlo.sine %0 : tensor<complex<f64>>
  func.return %1 : tensor<complex<f64>>
  // CHECK-NEXT: tensor<complex<f64>>
  // CHECK-NEXT: [3.1657785132161682, 1.9596010414216063]
}
