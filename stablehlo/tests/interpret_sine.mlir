// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: sine_op_test_bf16
func.func @sine_op_test_bf16() -> tensor<11xbf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xbf16>
  %1 = stablehlo.sine %0 : tensor<11xbf16>
  func.return %1 : tensor<11xbf16>
  // CHECK-NEXT: tensor<11xbf16>
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: -0.000000e+00 : bf16
  // CHECK-NEXT: 8.398430e-01 : bf16
  // CHECK-NEXT: 1.245120e-01 : bf16
  // CHECK-NEXT: 1.000980e-01 : bf16
  // CHECK-NEXT: 9.689330e-04 : bf16
  // CHECK-NEXT: 0xFFC0 : bf16
  // CHECK-NEXT: 0xFFC0 : bf16
  // CHECK-NEXT: 0x7FFF : bf16
  // CHECK-NEXT: 9.183550e-41 : bf16
  // CHECK-NEXT: -9.183550e-41 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_f16
func.func @sine_op_test_f16() -> tensor<11xf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7C00, 0xFC00, 0x7FFF, 0x0001, 0x8001]> : tensor<11xf16>
  %1 = stablehlo.sine %0 : tensor<11xf16>
  func.return %1 : tensor<11xf16>
  // CHECK-NEXT: tensor<11xf16>
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: 8.413080e-01 : f16
  // CHECK-NEXT: 1.246950e-01 : f16
  // CHECK-NEXT: 9.979240e-02 : f16
  // CHECK-NEXT: 9.675020e-04 : f16
  // CHECK-NEXT: 0xFE00 : f16
  // CHECK-NEXT: 0xFE00 : f16
  // CHECK-NEXT: 0x7FFF : f16
  // CHECK-NEXT: 5.960460e-08 : f16
  // CHECK-NEXT: -5.960460e-08 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_f32
func.func @sine_op_test_f32() -> tensor<11xf32> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 0x00000001, 0x80000001]> : tensor<11xf32>
  %1 = stablehlo.sine %0 : tensor<11xf32>
  func.return %1 : tensor<11xf32>
  // CHECK-NEXT: tensor<11xf32>
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: -0.000000e+00 : f32
  // CHECK-NEXT: 0.841470957 : f32
  // CHECK-NEXT: 0.12467473 : f32
  // CHECK-NEXT: 0.0998334214 : f32
  // CHECK-NEXT: -8.74227765E-8 : f32
  // CHECK-NEXT: 0xFFC00000 : f32
  // CHECK-NEXT: 0xFFC00000 : f32
  // CHECK-NEXT: 0x7FFFFFFF : f32
  // CHECK-NEXT: 1.401300e-45 : f32
  // CHECK-NEXT: -1.401300e-45 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_f64
func.func @sine_op_test_f64() -> tensor<11xf64> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  %1 = stablehlo.sine %0 : tensor<11xf64>
  func.return %1 : tensor<11xf64>
  // CHECK-NEXT: tensor<11xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 0.8414709848078965 : f64
  // CHECK-NEXT: 0.124674733385{{[0-9]+}} : f64
  // CHECK-NEXT: 0.099833416646{{[0-9]+}} : f64
  // CHECK-NEXT: 1.224646799147{{[0-9]+}}E-16 : f64
  // CHECK-NEXT: 0xFFF8000000000000 : f64
  // CHECK-NEXT: 0xFFF8000000000000 : f64
  // CHECK-NEXT: 0x7FFFFFFFFFFFFFFF : f64
  // CHECK-NEXT: 4.940660e-324 : f64
  // CHECK-NEXT: -4.940660e-324 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_c64
func.func @sine_op_test_c64() -> tensor<2xcomplex<f32>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  %1 = stablehlo.sine %0 : tensor<2xcomplex<f32>>
  func.return %1 : tensor<2xcomplex<f32>>
  // CHECK-NEXT: tensor<2xcomplex<f32>>
  // CHECK-NEXT: [6.1169281 : f32, 0.427974522 : f32]
  // CHECK-NEXT: [-15.7901983 : f32, -42.1433716 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: sine_op_test_c128
func.func @sine_op_test_c128() -> tensor<2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  %1 = stablehlo.sine %0 : tensor<2xcomplex<f64>>
  func.return %1 : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [6.116928012369{{[0-9]+}} : f64, 0.427974534506{{[0-9]+}} : f64]
  // CHECK-NEXT: [-15.790198357309{{[0-9]+}} : f64, -42.143370741504{{[0-9]+}} : f64]
}
