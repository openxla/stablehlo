// RUN: stablehlo-interpreter --interpret -split-input-file %s | FileCheck %s

// CHECK-LABEL: Evaluated results of function: iota_op_test_si4_dim_0
func.func @iota_op_test_si4_dim_0() -> tensor<3x4xi4> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xi4>
  func.return %0 : tensor<3x4xi4>
  // CHECK-NEXT: tensor<3x4xi4>
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 2 : i4
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si4_dim_1
func.func @iota_op_test_si4_dim_1() -> tensor<3x4xi4> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xi4>
  func.return %0 : tensor<3x4xi4>
  // CHECK-NEXT: tensor<3x4xi4>
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 3 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 3 : i4
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 3 : i4
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si4_dim_0_overflow
func.func @iota_op_test_si4_dim_0_overflow() -> tensor<16x1xi4> {
  %0 = stablehlo.iota dim = 0 : tensor<16x1xi4>
  func.return %0 : tensor<16x1xi4>
  // CHECK-NEXT: tensor<16x1xi4>
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 3 : i4
  // CHECK-NEXT: 4 : i4
  // CHECK-NEXT: 5 : i4
  // CHECK-NEXT: 6 : i4
  // CHECK-NEXT: 7 : i4
  // CHECK-NEXT: -8 : i4
  // CHECK-NEXT: -7 : i4
  // CHECK-NEXT: -6 : i4
  // CHECK-NEXT: -5 : i4
  // CHECK-NEXT: -4 : i4
  // CHECK-NEXT: -3 : i4
  // CHECK-NEXT: -2 : i4
  // CHECK-NEXT: -1 : i4
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si4_dim_1_overflow
func.func @iota_op_test_si4_dim_1_overflow() -> tensor<1x16xi4> {
  %0 = stablehlo.iota dim = 1 : tensor<1x16xi4>
  func.return %0 : tensor<1x16xi4>
  // CHECK-NEXT: tensor<1x16xi4>
  // CHECK-NEXT: 0 : i4
  // CHECK-NEXT: 1 : i4
  // CHECK-NEXT: 2 : i4
  // CHECK-NEXT: 3 : i4
  // CHECK-NEXT: 4 : i4
  // CHECK-NEXT: 5 : i4
  // CHECK-NEXT: 6 : i4
  // CHECK-NEXT: 7 : i4
  // CHECK-NEXT: -8 : i4
  // CHECK-NEXT: -7 : i4
  // CHECK-NEXT: -6 : i4
  // CHECK-NEXT: -5 : i4
  // CHECK-NEXT: -4 : i4
  // CHECK-NEXT: -3 : i4
  // CHECK-NEXT: -2 : i4
  // CHECK-NEXT: -1 : i4
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si8_dim_0
func.func @iota_op_test_si8_dim_0() -> tensor<3x4xi8> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xi8>
  func.return %0 : tensor<3x4xi8>
  // CHECK-NEXT: tensor<3x4xi8>
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 2 : i8
  // CHECK-NEXT: 2 : i8
  // CHECK-NEXT: 2 : i8
  // CHECK-NEXT: 2 : i8
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si8_dim_1
func.func @iota_op_test_si8_dim_1() -> tensor<3x4xi8> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xi8>
  func.return %0 : tensor<3x4xi8>
  // CHECK-NEXT: tensor<3x4xi8>
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 2 : i8
  // CHECK-NEXT: 3 : i8
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 2 : i8
  // CHECK-NEXT: 3 : i8
  // CHECK-NEXT: 0 : i8
  // CHECK-NEXT: 1 : i8
  // CHECK-NEXT: 2 : i8
  // CHECK-NEXT: 3 : i8
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si16_dim_0
func.func @iota_op_test_si16_dim_0() -> tensor<3x4xi16> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xi16>
  func.return %0 : tensor<3x4xi16>
  // CHECK-NEXT: tensor<3x4xi16>
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 2 : i16
  // CHECK-NEXT: 2 : i16
  // CHECK-NEXT: 2 : i16
  // CHECK-NEXT: 2 : i16
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si16_dim_1
func.func @iota_op_test_si16_dim_1() -> tensor<3x4xi16> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xi16>
  func.return %0 : tensor<3x4xi16>
  // CHECK-NEXT: tensor<3x4xi16>
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 2 : i16
  // CHECK-NEXT: 3 : i16
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 2 : i16
  // CHECK-NEXT: 3 : i16
  // CHECK-NEXT: 0 : i16
  // CHECK-NEXT: 1 : i16
  // CHECK-NEXT: 2 : i16
  // CHECK-NEXT: 3 : i16
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si32_dim_0
func.func @iota_op_test_si32_dim_0() -> tensor<3x4xi32> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xi32>
  func.return %0 : tensor<3x4xi32>
  // CHECK-NEXT: tensor<3x4xi32>
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 2 : i32
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si32_dim_1
func.func @iota_op_test_si32_dim_1() -> tensor<3x4xi32> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xi32>
  func.return %0 : tensor<3x4xi32>
  // CHECK-NEXT: tensor<3x4xi32>
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 3 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 3 : i32
  // CHECK-NEXT: 0 : i32
  // CHECK-NEXT: 1 : i32
  // CHECK-NEXT: 2 : i32
  // CHECK-NEXT: 3 : i32
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si64_dim_0
func.func @iota_op_test_si64_dim_0() -> tensor<3x4xi64> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
  // CHECK-NEXT: tensor<3x4xi64>
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 2 : i64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_si64_dim_1
func.func @iota_op_test_si64_dim_1() -> tensor<3x4xi64> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
  // CHECK-NEXT: tensor<3x4xi64>
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
  // CHECK-NEXT: 0 : i64
  // CHECK-NEXT: 1 : i64
  // CHECK-NEXT: 2 : i64
  // CHECK-NEXT: 3 : i64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_ui64_dim_0
func.func @iota_op_test_ui64_dim_0() -> tensor<3x4x5x3xui64> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4x5x3xui64>
  func.return %0 : tensor<3x4x5x3xui64>
  // CHECK-NEXT: tensor<3x4x5x3xui64>
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_ui64_dim_1
func.func @iota_op_test_ui64_dim_1() -> tensor<3x4x5x3xui64> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4x5x3xui64>
  func.return %0 : tensor<3x4x5x3xui64>
  // CHECK-NEXT: tensor<3x4x5x3xui64>
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_ui64_dim_2
func.func @iota_op_test_ui64_dim_2() -> tensor<3x4x5x3xui64> {
  %0 = stablehlo.iota dim = 2 : tensor<3x4x5x3xui64>
  func.return %0 : tensor<3x4x5x3xui64>
  // CHECK-NEXT: tensor<3x4x5x3xui64>
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 3 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
  // CHECK-NEXT: 4 : ui64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_ui64_dim_3
func.func @iota_op_test_ui64_dim_3() -> tensor<3x4x5x3xui64> {
  %0 = stablehlo.iota dim = 3 : tensor<3x4x5x3xui64>
  func.return %0 : tensor<3x4x5x3xui64>
  // CHECK-NEXT: tensor<3x4x5x3xui64>
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
  // CHECK-NEXT: 0 : ui64
  // CHECK-NEXT: 1 : ui64
  // CHECK-NEXT: 2 : ui64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_bf16_dim_0
func.func @iota_op_test_bf16_dim_0() -> tensor<3x4xbf16> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xbf16>
  func.return %0 : tensor<3x4xbf16>
  // CHECK-NEXT: tensor<3x4xbf16>
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_bf16_dim_1
func.func @iota_op_test_bf16_dim_1() -> tensor<3x4xbf16> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xbf16>
  func.return %0 : tensor<3x4xbf16>
  // CHECK-NEXT: tensor<3x4xbf16>
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
  // CHECK-NEXT: 3.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
  // CHECK-NEXT: 3.000000e+00 : bf16
  // CHECK-NEXT: 0.000000e+00 : bf16
  // CHECK-NEXT: 1.000000e+00 : bf16
  // CHECK-NEXT: 2.000000e+00 : bf16
  // CHECK-NEXT: 3.000000e+00 : bf16
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_f16_dim_0
func.func @iota_op_test_f16_dim_0() -> tensor<3x4xf16> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xf16>
  func.return %0 : tensor<3x4xf16>
  // CHECK-NEXT: tensor<3x4xf16>
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_f16_dim_1
func.func @iota_op_test_f16_dim_1() -> tensor<3x4xf16> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xf16>
  func.return %0 : tensor<3x4xf16>
  // CHECK-NEXT: tensor<3x4xf16>
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
  // CHECK-NEXT: 3.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
  // CHECK-NEXT: 3.000000e+00 : f16
  // CHECK-NEXT: 0.000000e+00 : f16
  // CHECK-NEXT: 1.000000e+00 : f16
  // CHECK-NEXT: 2.000000e+00 : f16
  // CHECK-NEXT: 3.000000e+00 : f16
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_f32_dim_0
func.func @iota_op_test_f32_dim_0() -> tensor<3x4xf32> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
  // CHECK-NEXT: tensor<3x4xf32>
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_f32_dim_1
func.func @iota_op_test_f32_dim_1() -> tensor<3x4xf32> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
  // CHECK-NEXT: tensor<3x4xf32>
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
  // CHECK-NEXT: 3.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
  // CHECK-NEXT: 3.000000e+00 : f32
  // CHECK-NEXT: 0.000000e+00 : f32
  // CHECK-NEXT: 1.000000e+00 : f32
  // CHECK-NEXT: 2.000000e+00 : f32
  // CHECK-NEXT: 3.000000e+00 : f32
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_f64_dim_0
func.func @iota_op_test_f64_dim_0() -> tensor<3x4xf64> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xf64>
  func.return %0 : tensor<3x4xf64>
  // CHECK-NEXT: tensor<3x4xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_f64_dim_1
func.func @iota_op_test_f64_dim_1() -> tensor<3x4xf64> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xf64>
  func.return %0 : tensor<3x4xf64>
  // CHECK-NEXT: tensor<3x4xf64>
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 3.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 3.000000e+00 : f64
  // CHECK-NEXT: 0.000000e+00 : f64
  // CHECK-NEXT: 1.000000e+00 : f64
  // CHECK-NEXT: 2.000000e+00 : f64
  // CHECK-NEXT: 3.000000e+00 : f64
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_c64_dim_0
func.func @iota_op_test_c64_dim_0() -> tensor<3x4xcomplex<f32>> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xcomplex<f32>>
  func.return %0 : tensor<3x4xcomplex<f32>>
  // CHECK-NEXT: tensor<3x4xcomplex<f32>>
  // CHECK-NEXT: [0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_c64_dim_1
func.func @iota_op_test_c64_dim_1() -> tensor<3x4xcomplex<f32>> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xcomplex<f32>>
  func.return %0 : tensor<3x4xcomplex<f32>>
  // CHECK-NEXT: tensor<3x4xcomplex<f32>>
  // CHECK-NEXT: [0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [3.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [3.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [0.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [1.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [2.000000e+00 : f32, 0.000000e+00 : f32]
  // CHECK-NEXT: [3.000000e+00 : f32, 0.000000e+00 : f32]
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_c128_dim_0
func.func @iota_op_test_c128_dim_0() -> tensor<3x4xcomplex<f64>> {
  %0 = stablehlo.iota dim = 0 : tensor<3x4xcomplex<f64>>
  func.return %0 : tensor<3x4xcomplex<f64>>
  // CHECK-NEXT: tensor<3x4xcomplex<f64>>
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
}

// CHECK-LABEL: Evaluated results of function: iota_op_test_c128_dim_1
func.func @iota_op_test_c128_dim_1() -> tensor<3x4xcomplex<f64>> {
  %0 = stablehlo.iota dim = 1 : tensor<3x4xcomplex<f64>>
  func.return %0 : tensor<3x4xcomplex<f64>>
  // CHECK-NEXT: tensor<3x4xcomplex<f64>>
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [3.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [3.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [0.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [1.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [2.000000e+00 : f64, 0.000000e+00 : f64]
  // CHECK-NEXT: [3.000000e+00 : f64, 0.000000e+00 : f64]
}
