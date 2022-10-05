// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: cosine_op_test_bf16
func.func @cosine_op_test_bf16() -> tensor<11xbf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xbf16>
  %1 = stablehlo.cosine %0 : tensor<11xbf16>
  func.return %1 : tensor<11xbf16>
  // CHECK-NEXT: tensor<11xbf16>
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 5.390630e-01 : bf16
  // CHECK-NEXT: 9.921870e-01 : bf16
  // CHECK-NEXT: 9.960930e-01 : bf16
  // CHECK-NEXT: -1.000000e+00 : bf16
  // CHECK-NEXT: 0xFFC0 : bf16
  // CHECK-NEXT: 0xFFC0 : bf16
  // CHECK-NEXT: 0x7FFF : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: cosine_op_test_f16
func.func @cosine_op_test_f16() -> tensor<11xf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7C00, 0xFC00, 0x7FFF, 0x0001, 0x8001]> : tensor<11xf16>
  %1 = stablehlo.cosine %0 : tensor<11xf16>
  func.return %1 : tensor<11xf16>
  // CHECK-NEXT: tensor<11xf16>
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 5.405270e-01 : f16
  // CHECK-NEXT: 9.921870e-01 : f16
  // CHECK-NEXT: 9.951170e-01 : f16
  // CHECK-NEXT: -1.000000e+00 : f16
  // CHECK-NEXT: 0xFE00 : f16
  // CHECK-NEXT: 0xFE00 : f16
  // CHECK-NEXT: 0x7FFF : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: cosine_op_test_f32
func.func @cosine_op_test_f32() -> tensor<11xf32> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 0x00000001, 0x80000001]> : tensor<11xf32>
  %1 = stablehlo.cosine %0 : tensor<11xf32>
  func.return %1 : tensor<11xf32>
  // CHECK-NEXT: tensor<11xf32>
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 0.540302277 : f32
  // CHECK-NEXT: 0.992197692 : f32
  // CHECK-NEXT: 0.995004177 : f32
  // CHECK-NEXT: -1.000000e+00 : f32
  // CHECK-NEXT: 0xFFC00000 : f32
  // CHECK-NEXT: 0xFFC00000 : f32
  // CHECK-NEXT: 0x7FFFFFFF : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: cosine_op_test_f64
func.func @cosine_op_test_f64() -> tensor<11xf64> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  %1 = stablehlo.cosine %0 : tensor<11xf64>
  func.return %1 : tensor<11xf64>
  // CHECK-NEXT: tensor<11xf64>
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 0.540302305868{{[0-9]+}} : f64
  // CHECK-NEXT: 0.992197667229{{[0-9]+}} : f64
  // CHECK-NEXT: 0.995004165278{{[0-9]+}} : f64
  // CHECK-NEXT: -1.000000e+00 : f64
  // CHECK-NEXT: 0xFFF8000000000000 : f64
  // CHECK-NEXT: 0xFFF8000000000000 : f64
  // CHECK-NEXT: 0x7FFFFFFFFFFFFFFF : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: cosine_op_test_c64
func.func @cosine_op_test_c64() -> tensor<2xcomplex<f32>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  %1 = stablehlo.cosine %0 : tensor<2xcomplex<f32>>
  func.return %1 : tensor<2xcomplex<f32>>
  // CHECK-NEXT: tensor<2xcomplex<f32>>
  // CHECK-NEXT: [4.337810e-01 : f32, -6.03504848 : f32]
  // CHECK-NEXT: [-42.1537743 : f32, 15.7863016 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: cosine_op_test_c128
func.func @cosine_op_test_c128() -> tensor<2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.cosine %0 : tensor<2xcomplex<f64>>
  func.return %1 : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [0.433780997607{{[0-9]+}} : f64, -6.035048637766{{[0-9]+}} : f64]
  // CHECK-NEXT: [-42.153773835602{{[0-9]+}} : f64, 15.786301507647{{[0-9]+}} : f64]
}
