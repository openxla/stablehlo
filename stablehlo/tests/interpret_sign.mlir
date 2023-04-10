// RUN: stablehlo-translate --interpret -split-input-file %s

// CHECK-LABEL: Evaluated results of function: sign_op_test_si4
func.func @sign_op_test_si4() -> tensor<3xi4> {
  %operand = stablehlo.constant dense<[-7, 0, 7]> : tensor<3xi4>
  %result = stablehlo.sign %operand : tensor<3xi4>
  func.return %result : tensor<3xi4>
  // CHECK-NEXT: tensor<3xi4>
  // CHECK-NEXT: -1 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_si8
func.func @sign_op_test_si8() -> tensor<3xi8> {
  %operand = stablehlo.constant dense<[-127, 0, 127]> : tensor<3xi8>
  %result = stablehlo.sign %operand : tensor<3xi8>
  func.return %result : tensor<3xi8>
  // CHECK-NEXT: tensor<3xi8>
  // CHECK-NEXT: -1 : i8
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 1 : i8
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_si16
func.func @sign_op_test_si16() -> tensor<3xi16> {
  %operand = stablehlo.constant dense<[-32767, 0, 32767]> : tensor<3xi16>
  %result = stablehlo.sign %operand : tensor<3xi16>
  func.return %result : tensor<3xi16>
  // CHECK-NEXT: tensor<3xi16>
  // CHECK-NEXT: -1 : i16
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 1 : i16
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_si32
func.func @sign_op_test_si32() -> tensor<3xi32> {
  %operand = stablehlo.constant dense<[-2147483647, 0, 2147483647]> : tensor<3xi32>
  %result = stablehlo.sign %operand : tensor<3xi32>
  func.return %result : tensor<3xi32>
  // CHECK-NEXT: tensor<3xi32>
  // CHECK-NEXT: -1 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 1 : i32
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_si64
func.func @sign_op_test_si64() -> tensor<3xi64> {
  %operand = stablehlo.constant dense<[-9223372036854775807, 0, 9223372036854775807]> : tensor<3xi64>
  %result = stablehlo.sign %operand : tensor<3xi64>
  func.return %result : tensor<3xi64>
  // CHECK-NEXT: tensor<3xi64>
  // CHECK-NEXT: -1 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_bf16
func.func @sign_op_test_bf16() -> tensor<8xbf16> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.140625, 3.140625, 0xFFFF, 0x7FFF, 0xFF80, 0x7F80]> : tensor<8xbf16>
  %result = stablehlo.sign %operand : tensor<8xbf16>
  func.return %result : tensor<8xbf16>
  // CHECK-NEXT: tensor<8xbf16>
  // CHECK-NEXT: -0.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: -1.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 0xFFFF : bf16
  // CHECK-NEXT: 0x7FFF : bf16
  // CHECK-NEXT: -1.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_f16
func.func @sign_op_test_f16() -> tensor<6xf16> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.141, 3.141, 0xFFFF, 0x7FFF]> : tensor<6xf16>
  %result = stablehlo.sign %operand : tensor<6xf16>
  func.return %result : tensor<6xf16>
  // CHECK-NEXT: tensor<6xf16>
  // CHECK-NEXT: -0.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: -1.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 0xFFFF : f16
  // CHECK-NEXT: 0x7FFF : f16
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_f32
func.func @sign_op_test_f32() -> tensor<8xf32> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.14159265, 3.14159265, 0xFFFFFFFF, 0x7FFFFFFF, 0xFF800000, 0x7F800000]> : tensor<8xf32>
  %result = stablehlo.sign %operand : tensor<8xf32>
  func.return %result : tensor<8xf32>
  // CHECK-NEXT: tensor<8xf32>
  // CHECK-NEXT: -0.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: -1.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 0xFFFFFFFF : f32
  // CHECK-NEXT: 0x7FFFFFFF : f32
  // CHECK-NEXT: -1.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_f64
func.func @sign_op_test_f64() -> tensor<8xf64> {
  // -0.0, +0.0, -3.14, +3.14, -NaN, +NaN, -Inf, +Inf
  %operand = stablehlo.constant dense<[-0.0, 0.0, -3.14159265358979323846, 3.14159265358979323846, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0xFFF0000000000000, 0x7FF0000000000000]> : tensor<8xf64>
  %result = stablehlo.sign %operand : tensor<8xf64>
  func.return %result : tensor<8xf64>
  // CHECK-NEXT: tensor<8xf64>
  // CHECK-NEXT: -0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: -1.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 0xFFFFFFFFFFFFFFFF : f64
  // CHECK-NEXT: 0x7FFFFFFFFFFFFFFF : f64
  // CHECK-NEXT: -1.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_c64
func.func @sign_op_test_c64() -> tensor<3xcomplex<f32>> {
  // (+NaN, +0.0), (+0.0, +NaN), (-1.0, 0.0)
  // (+NaN, +0.0), (+0.0, +NaN), (1.0, 1.0)
  %operand = stablehlo.constant dense<[(0x7FFFFFFF, 0.0), (0.0, 0x7FFFFFFF), (-1.0, 0.0)]> : tensor<3xcomplex<f32>>
  %result = stablehlo.sign %operand : tensor<3xcomplex<f32>>
  func.return %result : tensor<3xcomplex<f32>>
  // CHECK-NEXT: tensor<3xcomplex<f32>>
  // CHECK-NEXT: [0x7FFFFFFF : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [0x7FFFFFFF : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [-1.000000e+00 : f32, 0.000000e+00 : f32]
}

// -----

// CHECK-LABEL: Evaluated results of function: sign_op_test_c128
func.func @sign_op_test_c128() -> tensor<3xcomplex<f64>> {
  // (+NaN, +0.0), (+0.0, +NaN), (0.0, 1.0)
  // (+NaN, +0.0), (+Nan, +0.0), (1.0, 1.0)
  %operand = stablehlo.constant dense<[(0x7FF0000000000001, 0x0000000000000000), (0x0000000000000000, 0x7FF0000000000001), (0.0, 1.0)]> : tensor<3xcomplex<f64>>
  %result = stablehlo.sign %operand : tensor<3xcomplex<f64>>
  func.return %result : tensor<3xcomplex<f64>>
  // CHECK-NEXT: tensor<3xcomplex<f64>>
  // CHECK-NEXT: [0x7FF0000000000001, 0x0000000000000000]
  // CHECK-NEXT: [0x7FF0000000000001, 0x0000000000000000]
  // CHECK-NEXT: [0.000000e+00, 1.000000e+00]
}