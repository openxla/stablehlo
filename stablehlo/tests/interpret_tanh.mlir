// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: tanh_op_test_bf16
func.func @tanh_op_test_bf16() -> tensor<11xbf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xbf16>
  %1 = stablehlo.tanh %0 : tensor<11xbf16>
  func.return %1 : tensor<11xbf16>
  // CHECK-NEXT: tensor<11xbf16>
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: -0.000000e+00 : bf16
  // CHECK-NEXT: 7.617180e-01 : bf16
  // CHECK-NEXT: 1.245120e-01 : bf16
  // CHECK-NEXT: 9.960930e-02 : bf16
  // CHECK-NEXT: 9.960930e-01 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: -1.000000e+00 : bf16
  // CHECK-NEXT: 0x7FFF : bf16
  // CHECK-NEXT: 9.183550e-41 : bf16
  // CHECK-NEXT: -9.183550e-41 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: tanh_op_test_f16
func.func @tanh_op_test_f16() -> tensor<11xf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7C00, 0xFC00, 0x7FFF, 0x0001, 0x8001]> : tensor<11xf16>
  %1 = stablehlo.tanh %0 : tensor<11xf16>
  func.return %1 : tensor<11xf16>
  // CHECK-NEXT: tensor<11xf16>
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: 7.617180e-01 : f16
  // CHECK-NEXT: 1.243290e-01 : f16
  // CHECK-NEXT: 9.967040e-02 : f16
  // CHECK-NEXT: 9.960930e-01 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: -1.000000e+00 : f16
  // CHECK-NEXT: 0x7FFF : f16
  // CHECK-NEXT: 5.960460e-08 : f16
  // CHECK-NEXT: -5.960460e-08 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: tanh_op_test_f32
func.func @tanh_op_test_f32() -> tensor<11xf32> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 0x00000001, 0x80000001]> : tensor<11xf32>
  %1 = stablehlo.tanh %0 : tensor<11xf32>
  func.return %1 : tensor<11xf32>
  // CHECK-NEXT: tensor<11xf32>
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: -0.000000e+00 : f32
  // CHECK-NEXT: 0.761594176 : f32
  // CHECK-NEXT: 1.243530e-01 : f32
  // CHECK-NEXT: 0.0996679961 : f32
  // CHECK-NEXT: 0.996272087 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: -1.000000e+00 : f32
  // CHECK-NEXT: 0x7FFFFFFF : f32
  // CHECK-NEXT: 1.401300e-45 : f32
  // CHECK-NEXT: -1.401300e-45 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: tanh_op_test_f64
func.func @tanh_op_test_f64() -> tensor<11xf64> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  %1 = stablehlo.tanh %0 : tensor<11xf64>
  func.return %1 : tensor<11xf64>
  // CHECK-NEXT: tensor<11xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 0.761594155955{{[0-9]+}} : f64
  // CHECK-NEXT: 0.124353001771{{[0-9]+}} : f64
  // CHECK-NEXT: 0.099667994624{{[0-9]+}} : f64
  // CHECK-NEXT: 0.996272076220{{[0-9]+}} : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: -1.000000e+00 : f64
  // CHECK-NEXT: 0x7FFFFFFFFFFFFFFF : f64
  // CHECK-NEXT: 4.940660e-324 : f64
  // CHECK-NEXT: -4.940660e-324 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: tanh_op_test_c64
func.func @tanh_op_test_c64() -> tensor<2xcomplex<f32>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  %1 = stablehlo.tanh %0 : tensor<2xcomplex<f32>>
  func.return %1 : tensor<2xcomplex<f32>>
  // CHECK-NEXT: tensor<2xcomplex<f32>>
  // CHECK-NEXT: [0.967786788 : f32, -0.0926378369 : f32]
  // CHECK-NEXT: [1.00166273 : f32, 7.52857188E-4 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: tanh_op_test_c128
func.func @tanh_op_test_c128() -> tensor<2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.tanh %0 : tensor<2xcomplex<f64>>
  func.return %1 : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [0.967786802152{{[0-9]+}} : f64, -0.092637836268{{[0-9]+}} : f64]
  // CHECK-NEXT: [1.001662785095{{[0-9]+}} : f64, 7.528572153821{{[0-9]+}}E-4 : f64]
}
