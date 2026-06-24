// RUN: stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file %s > %t.mlir
// RUN: stablehlo-translate --interpret --split-input-file %t.mlir

// Test chlo.mulhi operation for signed 8-bit integers.
// 64*4 = 256 ==> b0000_00001_0000_0000, high bits are 1.
func.func @mulhi_op_test_si8() {
  %lhs = stablehlo.constant dense<[64, -64, -64]> : tensor<3xi8>
  %rhs = stablehlo.constant dense<[4, 4, -4]> : tensor<3xi8>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<3xi8>, tensor<3xi8>) -> tensor<3xi8>
  check.expect_eq_const %result, dense<[1, -1, 1]> : tensor<3xi8>
  func.return
}

// -----

// Test chlo.mulhi operation for unsigned 8-bit integers.
// 240 * 16 = 3840 ==> b0000_1111_0000_0000, high bits are 15.
func.func @mulhi_op_test_ui8() {
  %lhs = stablehlo.constant dense<[240, 128]> : tensor<2xui8>
  %rhs = stablehlo.constant dense<[16, 2]> : tensor<2xui8>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xui8>
  check.expect_eq_const %result, dense<[15, 1]> : tensor<2xui8>
  func.return
}

// -----

// Test chlo.mulhi operation for signed 16-bit integers.
func.func @mulhi_op_test_si16() {
  %lhs = stablehlo.constant dense<[16384, -16384]> : tensor<2xi16>
  %rhs = stablehlo.constant dense<[8, 8]> : tensor<2xi16>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<2xi16>, tensor<2xi16>) -> tensor<2xi16>
  check.expect_eq_const %result, dense<[2, -2]> : tensor<2xi16>
  func.return
}

// -----

// Test chlo.mulhi operation for unsigned 16-bit integers.
func.func @mulhi_op_test_ui16() {
  %lhs = stablehlo.constant dense<[61440, 32768]> : tensor<2xui16>
  %rhs = stablehlo.constant dense<[16, 2]> : tensor<2xui16>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<2xui16>, tensor<2xui16>) -> tensor<2xui16>
  check.expect_eq_const %result, dense<[15, 1]> : tensor<2xui16>
  func.return
}

// -----

// Test chlo.mulhi operation for signed 32-bit integers.
func.func @mulhi_op_test_si32() {
  %lhs = stablehlo.constant dense<[268435456, -268435456]> : tensor<2xi32>
  %rhs = stablehlo.constant dense<[16, 16]> : tensor<2xi32>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  check.expect_eq_const %result, dense<[1, -1]> : tensor<2xi32>
  func.return
}

// -----

// Test chlo.mulhi operation for unsigned 32-bit integers.
func.func @mulhi_op_test_ui32() {
  %lhs = stablehlo.constant dense<[4026531840, 2147483648]> : tensor<2xui32>
  %rhs = stablehlo.constant dense<[16, 2]> : tensor<2xui32>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
  check.expect_eq_const %result, dense<[15, 1]> : tensor<2xui32>
  func.return
}

// -----

// Test chlo.mulhi operation for signed 64-bit integers.
// 1152921504606846976 * 16 = 18446744073709551616 ==> b1_0{64_zeros}
func.func @mulhi_op_test_si64() {
  %lhs = stablehlo.constant dense<[1152921504606846976, -1152921504606846976]> : tensor<2xi64>
  %rhs = stablehlo.constant dense<[16, 16]> : tensor<2xi64>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  check.expect_eq_const %result, dense<[1, -1]> : tensor<2xi64>
  func.return
}

// -----

// Test chlo.mulhi operation for unsigned 64-bit integers.
func.func @mulhi_op_test_ui64() {
  %lhs = stablehlo.constant dense<[17293822569102704640, 9223372036854775808]> : tensor<2xui64>
  %rhs = stablehlo.constant dense<[16, 2]> : tensor<2xui64>
  %result = "chlo.mulhi"(%lhs, %rhs) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
  check.expect_eq_const %result, dense<[15, 1]> : tensor<2xui64>
  func.return
}
