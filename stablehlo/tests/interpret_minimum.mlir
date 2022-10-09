// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: min_op_test_si4
func.func @min_op_test_si4() -> tensor<5xi4> {
  %0 = stablehlo.constant dense<[0, 1, 2, -3, 0]> : tensor<5xi4>
  %1 = stablehlo.constant dense<[-8, -1, 2, -3, 7]> : tensor<5xi4>
  %2 = stablehlo.minimum %0, %1 : tensor<5xi4>
  func.return %2 : tensor<5xi4>
  // CHECK-NEXT: tensor<5xi4>
  // CHECK-NEXT: -8 : i4
  // CHECK-NEXT: -1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: -3 : i4
  // CHECK-NEXT: 0 : i4
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_ui4
func.func @min_op_test_ui4() -> tensor<2xui4> {
  %0 = stablehlo.constant dense<[0, 2]> : tensor<2xui4>
  %1 = stablehlo.constant dense<[15, 3]> : tensor<2xui4>
  %2 = stablehlo.minimum %0, %1 : tensor<2xui4>
  func.return %2 : tensor<2xui4>
  // CHECK-NEXT: tensor<2xui4>
  // CHECK-NEXT: 0 : ui4
  // CHECK-NEXT: 2 : ui4
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_si8
func.func @min_op_test_si8() -> tensor<5xi8> {
  %0 = stablehlo.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi8>
  %1 = stablehlo.constant dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>
  %2 = stablehlo.minimum %0, %1 : tensor<5xi8>
  func.return %2 : tensor<5xi8>
  // CHECK-NEXT: tensor<5xi8>
  // CHECK-NEXT: -128 : i8
  // CHECK-NEXT: -1 : i8
  // CHECK-NEXT: 8 : i8
  // CHECK-NEXT: -9 : i8
  // CHECK-NEXT: 0 : i8
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_ui8
func.func @min_op_test_ui8() -> tensor<2xui8> {
  %0 = stablehlo.constant dense<[0, 16]> : tensor<2xui8>
  %1 = stablehlo.constant dense<[255, 16]> : tensor<2xui8>
  %2 = stablehlo.minimum %0, %1 : tensor<2xui8>
  func.return %2 : tensor<2xui8>
  // CHECK-NEXT: tensor<2xui8>
  // CHECK-NEXT: 0 : ui8
  // CHECK-NEXT: 16 : ui8
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_si16
func.func @min_op_test_si16() -> tensor<5xi16> {
  %0 = stablehlo.constant dense<[0, 1, 128, -129, 0]> : tensor<5xi16>
  %1 = stablehlo.constant dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>
  %2 = stablehlo.minimum %0, %1 : tensor<5xi16>
  func.return %2 : tensor<5xi16>
  // CHECK-NEXT: tensor<5xi16>
  // CHECK-NEXT: -32768 : i16
  // CHECK-NEXT: -1 : i16
  // CHECK-NEXT: 128 : i16
  // CHECK-NEXT: -129 : i16
  // CHECK-NEXT: 0 : i16
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_ui16
func.func @min_op_test_ui16() -> tensor<2xui16> {
  %0 = stablehlo.constant dense<[0, 256]> : tensor<2xui16>
  %1 = stablehlo.constant dense<[65535, 256]> : tensor<2xui16>
  %2 = stablehlo.minimum %0, %1 : tensor<2xui16>
  func.return %2 : tensor<2xui16>
  // CHECK-NEXT: tensor<2xui16>
  // CHECK-NEXT: 0 : ui16
  // CHECK-NEXT: 256 : ui16
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_si32
func.func @min_op_test_si32() -> tensor<5xi32> {
  %0 = stablehlo.constant dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>
  %1 = stablehlo.constant dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>
  %2 = stablehlo.minimum %0, %1 : tensor<5xi32>
  func.return %2 : tensor<5xi32>
  // CHECK-NEXT: tensor<5xi32>
  // CHECK-NEXT: -2147483648 : i32
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: 32768 : i32
  // CHECK-NEXT: -32769 : i32
  // CHECK-NEXT: 0 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_ui32
func.func @min_op_test_ui32() -> tensor<2xui32> {
  %0 = stablehlo.constant dense<[0, 65536]> : tensor<2xui32>
  %1 = stablehlo.constant dense<[4294967295, 65536]> : tensor<2xui32>
  %2 = stablehlo.minimum %0, %1 : tensor<2xui32>
  func.return %2 : tensor<2xui32>
  // CHECK-NEXT: tensor<2xui32>
  // CHECK-NEXT: 0 : ui32
  // CHECK-NEXT: 65536 : ui32
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_si64
func.func @min_op_test_si64() -> tensor<5xi64> {
  %0 = stablehlo.constant dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>
  %1 = stablehlo.constant dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>
  %2 = stablehlo.minimum %0, %1 : tensor<5xi64>
  func.return %2 : tensor<5xi64>
  // CHECK-NEXT: tensor<5xi64>
  // CHECK-NEXT: -9223372036854775808 : i64
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 2147483648 : i64
  // CHECK-NEXT: -2147483649 : i64
  // CHECK-NEXT: 0 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_ui64
func.func @min_op_test_ui64() -> tensor<2xui64> {
  %0 = stablehlo.constant dense<[0, 4294967296]> : tensor<2xui64>
  %1 = stablehlo.constant dense<[18446744073709551615, 4294967296]> : tensor<2xui64>
  %2 = stablehlo.minimum %0, %1 : tensor<2xui64>
  func.return %2 : tensor<2xui64>
  // CHECK-NEXT: tensor<2xui64>
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 4294967296 : ui64
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_i1
func.func @min_op_test_i1() -> tensor<4xi1> {
  %0 = stablehlo.constant dense<[false, false, true, true]> : tensor<4xi1>
  %1 = stablehlo.constant dense<[false, true, false, true]> : tensor<4xi1>
  %2 = stablehlo.minimum %0, %1 : tensor<4xi1>
  func.return %2 : tensor<4xi1>
  // CHECK-NEXT: tensor<4xi1>
  // CHECK-NEXT: false
  // CHECK-NEXT: false
  // CHECK-NEXT: false
  // CHECK-NEXT: true
}

// -----


// CHECK-LABEL: Evaluated results of function: min_op_test_bf16
func.func @min_op_test_bf16() -> tensor<11xbf16> {
  %0 = stablehlo.constant dense<[0xFF80, 0xFF80, -1.0, 0x8001, 0.0, 0.0, 0x0001, 1.0, 0x7F80, 0x7F80, 0x7FC0]>  : tensor<11xbf16>
  %1 = stablehlo.constant dense<[0xFF80, -1.0, 0x8001, -0.0, -0.0, 0x0001, 1.0, 0x7F80, 0x7F80, 0xFF80, 0x7F80]> : tensor<11xbf16>
  %2 = stablehlo.minimum %0, %1 : tensor<11xbf16>
  func.return %2 : tensor<11xbf16>
  // CHECK-NEXT: tensor<11xbf16>
  // CHECK-NEXT: 0xFF80 : bf16
  // CHECK-NEXT: 0xFF80 : bf16
  // CHECK-NEXT: -1.000000e+00 : bf16
  // CHECK-NEXT: -9.183550e-41 : bf16
  // CHECK-NEXT: -0.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 9.183550e-41 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 0x7F80 : bf16
  // CHECK-NEXT: 0xFF80 : bf16
  // CHECK-NEXT: 0x7FC0 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_f16
func.func @min_op_test_f16() -> tensor<11xf16> {
  %0 = stablehlo.constant dense<[0xFC00, 0xFC00, -1.0, 0x8001, 0.0, 0.0, 0x0001, 1.0, 0x7C00, 0x7C00, 0x7E00]>  : tensor<11xf16>
  %1 = stablehlo.constant dense<[0xFC00, -1.0, 0x8001, -0.0, -0.0, 0x0001, 1.0, 0x7C00, 0x7C00, 0xFC00, 0x7C00]> : tensor<11xf16>
  %2 = stablehlo.minimum %0, %1 : tensor<11xf16>
  func.return %2 : tensor<11xf16>
  // CHECK-NEXT: tensor<11xf16>
  // CHECK-NEXT: 0xFC00 : f16
  // CHECK-NEXT: 0xFC00 : f16
  // CHECK-NEXT: -1.000000e+00  : f16
  // CHECK-NEXT: -5.960460e-08 : f16
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00  : f16
  // CHECK-NEXT: 5.960460e-08 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 0x7C00 : f16
  // CHECK-NEXT: 0xFC00 : f16
  // CHECK-NEXT: 0x7E00 : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_f32
func.func @min_op_test_f32() -> tensor<11xf32> {
  %0 = stablehlo.constant dense<[0xFF800000, 0xFF800000, -1.0, 0x80000001, 0.0, 0.0, 0x00000001, 1.0, 0x7F800000, 0x7F800000, 0x7FC00000]>  : tensor<11xf32>
  %1 = stablehlo.constant dense<[0xFF800000, -1.0, 0x80000001, -0.0, -0.0, 0x00000001, 1.0, 0x7F800000, 0x7F800000, 0xFF800000, 0x7F800000]> : tensor<11xf32>
  %2 = stablehlo.minimum %0, %1 : tensor<11xf32>
  func.return %2 : tensor<11xf32>
  // CHECK-NEXT: tensor<11xf32>
  // CHECK-NEXT: 0xFF800000 : f32
  // CHECK-NEXT: 0xFF800000 : f32
  // CHECK-NEXT: -1.000000e+00 : f32
  // CHECK-NEXT: -1.401300e-45 : f32
  // CHECK-NEXT: -0.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 1.401300e-45 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 0x7F800000 : f32
  // CHECK-NEXT: 0xFF800000 : f32
  // CHECK-NEXT: 0x7FC00000 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_f64
func.func @min_op_test_f64() -> tensor<11xf64> {
  %0 = stablehlo.constant dense<[0xFFF0000000000000, 0xFFF0000000000000, -1.0, 0x8000000000000001, 0.0, 0.0, 0x0000000000000001, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0x7FF8000000000000]>  : tensor<11xf64>
  %1 = stablehlo.constant dense<[0xFFF0000000000000, -1.0, 0x8000000000000001, -0.0, -0.0, 0x0000000000000001, 1.0, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000000]> : tensor<11xf64>
  %2 = stablehlo.minimum %0, %1 : tensor<11xf64>
  func.return %2 : tensor<11xf64>
  // CHECK-NEXT: tensor<11xf64>
  // CHECK-NEXT: 0xFFF0000000000000 : f64
  // CHECK-NEXT: 0xFFF0000000000000 : f64
  // CHECK-NEXT: -1.000000e+00 : f64
  // CHECK-NEXT: -4.940660e-324 : f64
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 4.940660e-324 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 0x7FF0000000000000 : f64
  // CHECK-NEXT: 0xFFF0000000000000 : f64
  // CHECK-NEXT: 0x7FF8000000000000 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_c64
func.func @min_op_test_c64() -> tensor<4xcomplex<f32>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (1.5, 7.5), (0.0, 1.5), (0.0, 1.5)]> : tensor<4xcomplex<f32>>
  %1 = stablehlo.constant dense<[(7.5, 1.5), (1.5, 2.5), (-0.0, 2.5), (0.0, 1.5)]> : tensor<4xcomplex<f32>>
  %2 = stablehlo.minimum %0, %1 : tensor<4xcomplex<f32>>
  func.return %2 : tensor<4xcomplex<f32>>
  // CHECK-NEXT: tensor<4xcomplex<f32>> {
  // CHECK-NEXT: [1.500000e+00 : f32, 2.500000e+00 : f32]
  // CHECK-NEXT: [1.500000e+00 : f32, 2.500000e+00 : f32]
  // CHECK-NEXT: [0.000000e+00 : f32, 1.500000e+00 : f32]
  // CHECK-NEXT: [0.000000e+00 : f32, 1.500000e+00 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: min_op_test_c128
func.func @min_op_test_c128() -> tensor<4xcomplex<f64>> {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (1.5, 7.5), (0.0, 1.5), (0.0, 1.5)]> : tensor<4xcomplex<f64>>
  %1 = stablehlo.constant dense<[(7.5, 1.5), (1.5, 2.5), (-0.0, 2.5), (0.0, 1.5)]> : tensor<4xcomplex<f64>>
  %2 = stablehlo.minimum %0, %1 : tensor<4xcomplex<f64>>
  func.return %2 : tensor<4xcomplex<f64>>
  // CHECK-NEXT: tensor<4xcomplex<f64>> {
  // CHECK-NEXT: [1.500000e+00 : f64, 2.500000e+00 : f64]
  // CHECK-NEXT: [1.500000e+00 : f64, 2.500000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 1.500000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 1.500000e+00 : f64]
}
