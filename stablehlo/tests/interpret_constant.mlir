// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: constant_op_test_i32
func.func @constant_op_test_i32() -> tensor<2x3xi32> {
  %0 = stablehlo.constant dense<[[-1, -1, -1], [-1, -1, -1]]>: tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
  // CHECK-NEXT: tensor<2x3xi32>
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: -1 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_ui32
func.func @constant_op_test_ui32() -> tensor<2x3xui32> {
  %0 = stablehlo.constant dense<[[1, 1, 1], [1, 1, 1]]>: tensor<2x3xui32>
  func.return %0 : tensor<2x3xui32>
  // CHECK-NEXT: tensor<2x3xui32>
  // CHECK-NEXT: 1 : ui32
  // CHECK-NEXT: 1 : ui32
  // CHECK-NEXT: 1 : ui32
  // CHECK-NEXT: 1 : ui32
  // CHECK-NEXT: 1 : ui32
  // CHECK-NEXT: 1 : ui32
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_bf16
func.func @constant_op_test_bf16() -> tensor<bf16> {
  %0 = stablehlo.constant dense<1.0>: tensor<bf16>
  func.return %0 : tensor<bf16>
  // CHECK-NEXT: tensor<bf16>
  // CHECK-NEXT: 1.000000e+00 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_f16
func.func @constant_op_test_f16() -> tensor<f16> {
  %0 = stablehlo.constant dense<1.0>: tensor<f16>
  func.return %0 : tensor<f16>
  // CHECK-NEXT: tensor<f16>
  // CHECK-NEXT: 1.000000e+00 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_f32
func.func @constant_op_test_f32() -> tensor<f32> {
  %0 = stablehlo.constant dense<1.0>: tensor<f32>
  func.return %0 : tensor<f32>
  // CHECK-NEXT: tensor<f32>
  // CHECK-NEXT: 1.000000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_f64
func.func @constant_op_test_f64() -> tensor<2x3xf64> {
  %0 = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>: tensor<2x3xf64>
  func.return %0 : tensor<2x3xf64>
  // CHECK-NEXT: tensor<2x3xf64>
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 3.000000e+00 : f64
  // CHECK-NEXT: 4.000000e+00 : f64
  // CHECK-NEXT: 5.000000e+00 : f64
  // CHECK-NEXT: 6.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_c64
func.func @constant_op_test_c64() -> tensor<complex<f32>> {
  %0 = stablehlo.constant dense<(1.0, 2.0)>: tensor<complex<f32>>
  func.return %0 : tensor<complex<f32>>
  // CHECK-NEXT: tensor<complex<f32>>
  // CHECK-NEXT: [1.000000e+00 : f32, 2.000000e+00 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_c128
func.func @constant_op_test_c128() -> tensor<2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.0, 2.0), (3.0, 4.0)]>: tensor<2xcomplex<f64>>
  func.return %0 : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [1.000000e+00, 2.000000e+00]
  // CHECK-NEXT: [3.000000e+00, 4.000000e+00]
}
