// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: ceil_op_test_bf16
func.func @ceil_op_test_bf16() -> tensor<9xbf16> {
  %0 = stablehlo.constant dense<[0xFF80, -2.5, 0x8001, -0.0, 0.0, 0x0001, 2.5, 0x7F80, 0x7FC0]>  : tensor<9xbf16>
  %1 = stablehlo.ceil %0 : tensor<9xbf16>
  func.return %1 : tensor<9xbf16>
  // CHECK-NEXT: tensor<9xbf16> {
  // CHECK-NEXT: 0xFF80 : bf16
  // CHECK-NEXT: -2.000000e+00 : bf16
  // CHECK-NEXT: -0.000000e+00 : bf16
  // CHECK-NEXT: -0.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 3.000000e+00 : bf16
  // CHECK-NEXT: 0x7F80 : bf16
  // CHECK-NEXT: 0x7FC0 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: ceil_op_test_f16
func.func @ceil_op_test_f16() -> tensor<9xf16> {
  %0 = stablehlo.constant dense<[0xFC00, -2.5, 0x8001, -0.0, 0.0, 0x0001, 2.5, 0x7C00, 0x7E00]>  : tensor<9xf16>
  %1 = stablehlo.ceil %0 : tensor<9xf16>
  func.return %1 : tensor<9xf16>
  // CHECK-NEXT: tensor<9xf16>
  // CHECK-NEXT: 0xFC00 : f16
  // CHECK-NEXT: -2.000000e+00 : f16
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 3.000000e+00 : f16
  // CHECK-NEXT: 0x7C00 : f16
  // CHECK-NEXT: 0x7E00 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: ceil_op_test_f32
func.func @ceil_op_test_f32() -> tensor<9xf32> {
  %0 = stablehlo.constant dense<[0xFF800000, -2.5, 0x80000001, -0.0, 0.0, 0x00000001, 2.5, 0x7F800000, 0x7FC00000]>  : tensor<9xf32>
  %1 = stablehlo.ceil %0 : tensor<9xf32>
  func.return %1 : tensor<9xf32>
  // CHECK-NEXT: tensor<9xf32>
  // CHECK-NEXT: 0xFF800000 : f32
  // CHECK-NEXT: -2.000000e+00 : f32
  // CHECK-NEXT: -0.000000e+00 : f32
  // CHECK-NEXT: -0.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 3.000000e+00 : f32
  // CHECK-NEXT: 0x7F800000 : f32
  // CHECK-NEXT: 0x7FC00000 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: ceil_op_test_f64
func.func @ceil_op_test_f64() -> tensor<9xf64> {
  %0 = stablehlo.constant dense<[0xFFF0000000000000, -2.5, 0x8000000000000001, -0.0, 0.0, 0x0000000000000001, 2.5, 0x7FF0000000000000, 0x7FF8000000000000]>  : tensor<9xf64>
  %1 = stablehlo.ceil %0 : tensor<9xf64>
  func.return %1 : tensor<9xf64>
  // CHECK-NEXT: tensor<9xf64> {
  // CHECK-NEXT: 0xFFF0000000000000 : f64
  // CHECK-NEXT: -2.000000e+00 : f64
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 3.000000e+00 : f64
  // CHECK-NEXT: 0x7FF0000000000000 : f64
  // CHECK-NEXT: 0x7FF8000000000000 : f64
}
