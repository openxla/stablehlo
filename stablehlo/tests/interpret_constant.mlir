// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: constant_op_test_si4
func.func @constant_op_test_si4() -> tensor<5xi4> {
  %0 = stablehlo.constant dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>
  func.return %0 : tensor<5xi4>
  // CHECK-NEXT: tensor<5xi4>
  // CHECK-NEXT: -8 : i4
  // CHECK-NEXT: -1 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 7 : i4
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_ui4
func.func @constant_op_test_ui4() -> tensor<3xui4> {
  %0 = stablehlo.constant dense<[0, 8, 15]> : tensor<3xui4>
  func.return %0 : tensor<3xui4>
  // CHECK-NEXT: tensor<3xui4>
  // CHECK-NEXT: 0 : ui4
  // CHECK-NEXT: 8 : ui4
  // CHECK-NEXT: 15 : ui4
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_si8
func.func @constant_op_test_si8() -> tensor<5xi8> {
  %0 = stablehlo.constant dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>
  func.return %0 : tensor<5xi8>
  // CHECK-NEXT: tensor<5xi8>
  // CHECK-NEXT: -128 : i8
  // CHECK-NEXT: -9 : i8
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 8 : i8
  // CHECK-NEXT: 127 : i8
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_ui8
func.func @constant_op_test_ui8() -> tensor<3xui8> {
  %0 = stablehlo.constant dense<[0, 16, 255]> : tensor<3xui8>
  func.return %0 : tensor<3xui8>
  // CHECK-NEXT: tensor<3xui8>
  // CHECK-NEXT: 0 : ui8
  // CHECK-NEXT: 16 : ui8
  // CHECK-NEXT: 255 : ui8
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_si16
func.func @constant_op_test_si16() -> tensor<5xi16> {
  %0 = stablehlo.constant dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>
  func.return %0 : tensor<5xi16>
  // CHECK-NEXT: tensor<5xi16>
  // CHECK-NEXT: -32768 : i16
  // CHECK-NEXT: -129 : i16
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 128 : i16
  // CHECK-NEXT: 32767 : i16
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_ui16
func.func @constant_op_test_ui16() -> tensor<3xui16> {
  %0 = stablehlo.constant dense<[0, 256, 65535]> : tensor<3xui16>
  func.return %0 : tensor<3xui16>
  // CHECK-NEXT: tensor<3xui16>
  // CHECK-NEXT: 0 : ui16
  // CHECK-NEXT: 256 : ui16
  // CHECK-NEXT: 65535 : ui16
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_si32
func.func @constant_op_test_si32() -> tensor<5xi32> {
  %0 = stablehlo.constant dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>
  func.return %0 : tensor<5xi32>
  // CHECK-NEXT: tensor<5xi32>
  // CHECK-NEXT: -2147483648 : i32
  // CHECK-NEXT: -65537 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 65536 : i32
  // CHECK-NEXT: 2147483647 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_ui32
func.func @constant_op_test_ui32() -> tensor<3xui32> {
  %0 = stablehlo.constant dense<[0, 65536, 4294967295]> : tensor<3xui32>
  func.return %0 : tensor<3xui32>
  // CHECK-NEXT: tensor<3xui32>
  // CHECK-NEXT: 0 : ui32
  // CHECK-NEXT: 65536 : ui32
  // CHECK-NEXT: 4294967295 : ui32
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_si64
func.func @constant_op_test_si64() -> tensor<5xi64> {
  %0 = stablehlo.constant dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>
  func.return %0 : tensor<5xi64>
  // CHECK-NEXT: tensor<5xi64>
  // CHECK-NEXT: -9223372036854775808 : i64
  // CHECK-NEXT: -2147483649 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 2147483648 : i64
  // CHECK-NEXT: 9223372036854775807 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_ui64
func.func @constant_op_test_ui64() -> tensor<3xui64> {
  %0 = stablehlo.constant dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>
  func.return %0 : tensor<3xui64>
  // CHECK-NEXT: tensor<3xui64>
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 4294967296 : ui64
  // CHECK-NEXT: 18446744073709551615 : ui64
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_bf16
func.func @constant_op_test_bf16() -> tensor<11xbf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xbf16>
  func.return %0 : tensor<11xbf16>
  // CHECK-NEXT: tensor<11xbf16>
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: -0.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 1.250000e-01 : bf16
  // CHECK-NEXT: 1.000980e-01 : bf16
  // CHECK-NEXT: 3.140630e+00 : bf16
  // CHECK-NEXT: 0x7F80 : bf16
  // CHECK-NEXT: 0xFF80 : bf16
  // CHECK-NEXT: 0x7FFF : bf16
  // CHECK-NEXT: 9.183550e-41 : bf16
  // CHECK-NEXT: -9.183550e-41 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_f16
func.func @constant_op_test_f16() -> tensor<11xf16> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7C00, 0xFC00, 0x7FFF, 0x0001, 0x8001]> : tensor<11xf16>
  func.return %0 : tensor<11xf16>
  // CHECK-NEXT: tensor<11xf16>
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 1.250000e-01 : f16
  // CHECK-NEXT: 9.997550e-02 : f16
  // CHECK-NEXT: 3.140630e+00 : f16
  // CHECK-NEXT: 0x7C00 : f16
  // CHECK-NEXT: 0xFC00 : f16
  // CHECK-NEXT: 0x7FFF : f16
  // CHECK-NEXT: 5.960460e-08 : f16
  // CHECK-NEXT: -5.960460e-08 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_f32
func.func @constant_op_test_f32() -> tensor<11xf32> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 0x00000001, 0x80000001]> : tensor<11xf32>
  func.return %0 : tensor<11xf32>
  // CHECK-NEXT: tensor<11xf32>
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: -0.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 1.250000e-01 : f32
  // CHECK-NEXT: 1.000000e-01 : f32
  // CHECK-NEXT: 3.14159274 : f32
  // CHECK-NEXT: 0x7F800000 : f32
  // CHECK-NEXT: 0xFF800000 : f32
  // CHECK-NEXT: 0x7FFFFFFF : f32
  // CHECK-NEXT: 1.401300e-45 : f32
  // CHECK-NEXT: -1.401300e-45 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_f64
func.func @constant_op_test_f64() -> tensor<11xf64> {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  func.return %0 : tensor<11xf64>
  // CHECK-NEXT: tensor<11xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 1.250000e-01 : f64
  // CHECK-NEXT: 1.000000e-01 : f64
  // CHECK-NEXT: 3.1415926535897931 : f64
  // CHECK-NEXT: 0x7FF0000000000000 : f64
  // CHECK-NEXT: 0xFFF0000000000000 : f64
  // CHECK-NEXT: 0x7FFFFFFFFFFFFFFF : f64
  // CHECK-NEXT: 4.940660e-324 : f64
  // CHECK-NEXT: -4.940660e-324 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_c64
func.func @constant_op_test_c64() -> tensor<2xcomplex<f32>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
  // CHECK-NEXT: tensor<2xcomplex<f32>>
  // CHECK-NEXT: [1.500000e+00 : f32, 2.500000e+00 : f32]
  // CHECK-NEXT: [3.500000e+00 : f32, 4.500000e+00 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: constant_op_test_c128
func.func @constant_op_test_c128() -> tensor<2xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  func.return %0 : tensor<2xcomplex<f64>>
  // CHECK-NEXT: tensor<2xcomplex<f64>>
  // CHECK-NEXT: [1.500000e+00 : f64, 2.500000e+00 : f64]
  // CHECK-NEXT: [3.500000e+00 : f64, 4.500000e+00 : f64]
}
