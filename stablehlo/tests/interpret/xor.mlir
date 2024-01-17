// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @xor_op_test_si4() {
  %0 = stablehlo.constant dense<[7, -8, -8]> : tensor<3xi4>
  %1 = stablehlo.constant dense<[0, 7, -8]> : tensor<3xi4>
  %2 = stablehlo.xor %0, %1 : tensor<3xi4>
  check.expect_eq_const %2, dense<[7, -1, 0]> : tensor<3xi4>
  func.return
}

// -----

func.func @xor_op_test_ui4() {
  %0 = stablehlo.constant dense<[0, 7, 15]> : tensor<3xui4>
  %1 = stablehlo.constant dense<15> : tensor<3xui4>
  %2 = stablehlo.xor %0, %1 : tensor<3xui4>
  check.expect_eq_const %2, dense<[15, 8, 0]> : tensor<3xui4>
  func.return
}

// -----

func.func @xor_op_test_si8() {
  %0 = stablehlo.constant dense<[127, -128, -128]> : tensor<3xi8>
  %1 = stablehlo.constant dense<[0, 127, -128]> : tensor<3xi8>
  %2 = stablehlo.xor %0, %1 : tensor<3xi8>
  check.expect_eq_const %2, dense<[127, -1, 0]> : tensor<3xi8>
  func.return
}

// -----

func.func @xor_op_test_ui8() {
  %0 = stablehlo.constant dense<[0, 127, 255]> : tensor<3xui8>
  %1 = stablehlo.constant dense<255> : tensor<3xui8>
  %2 = stablehlo.xor %0, %1 : tensor<3xui8>
  check.expect_eq_const %2, dense<[255, 128, 0]> : tensor<3xui8>
  func.return
}

// -----

func.func @xor_op_test_si16() {
  %0 = stablehlo.constant dense<[32767, -32768, -32768]> : tensor<3xi16>
  %1 = stablehlo.constant dense<[0, 32767, -32768]> : tensor<3xi16>
  %2 = stablehlo.xor %0, %1 : tensor<3xi16>
  check.expect_eq_const %2, dense<[32767, -1, 0]> : tensor<3xi16>
  func.return
}

// -----

func.func @xor_op_test_ui16() {
  %0 = stablehlo.constant dense<[0, 32767, 65535]> : tensor<3xui16>
  %1 = stablehlo.constant dense<65535> : tensor<3xui16>
  %2 = stablehlo.xor %0, %1 : tensor<3xui16>
  check.expect_eq_const %2, dense<[65535, 32768, 0]> : tensor<3xui16>
  func.return
}

// -----

func.func @xor_op_test_si32() {
  %0 = stablehlo.constant dense<[2147483647, -2147483648, -2147483648]> : tensor<3xi32>
  %1 = stablehlo.constant dense<[0, 2147483647, -2147483648]> : tensor<3xi32>
  %2 = stablehlo.xor %0, %1 : tensor<3xi32>
  check.expect_eq_const %2, dense<[2147483647, -1, 0]> : tensor<3xi32>
  func.return
}

// -----

func.func @xor_op_test_ui32() {
  %0 = stablehlo.constant dense<[0, 2147483647, 4294967295]> : tensor<3xui32>
  %1 = stablehlo.constant dense<4294967295> : tensor<3xui32>
  %2 = stablehlo.xor %0, %1 : tensor<3xui32>
  check.expect_eq_const %2, dense<[4294967295, 2147483648, 0]> : tensor<3xui32>
  func.return
}

// -----

func.func @xor_op_test_si64() {
  %0 = stablehlo.constant dense<[9223372036854775807, -9223372036854775808, -9223372036854775808]> : tensor<3xi64>
  %1 = stablehlo.constant dense<[0, 9223372036854775807, -9223372036854775808]> : tensor<3xi64>
  %2 = stablehlo.xor %0, %1 : tensor<3xi64>
  check.expect_eq_const %2, dense<[9223372036854775807, -1, 0]> : tensor<3xi64>
  func.return
}

// -----

func.func @xor_op_test_ui64() {
  %0 = stablehlo.constant dense<[0, 9223372036854775807, 18446744073709551615]> : tensor<3xui64>
  %1 = stablehlo.constant dense<18446744073709551615> : tensor<3xui64>
  %2 = stablehlo.xor %0, %1 : tensor<3xui64>
  check.expect_eq_const %2, dense<[18446744073709551615, 9223372036854775808, 0]> : tensor<3xui64>
  func.return
}

// -----

func.func @xor_op_test_i1() {
  %0 = stablehlo.constant dense<[false, false, true, true]> : tensor<4xi1>
  %1 = stablehlo.constant dense<[false, true, false, true]> : tensor<4xi1>
  %2 = stablehlo.xor %0, %1 : tensor<4xi1>
  check.expect_eq_const %2, dense<[false, true, true, false]> : tensor<4xi1>
  func.return
}

// -----

func.func @xor_op_test_i1_splat_false() {
  %0 = stablehlo.constant dense<false> : tensor<2xi1>
  %1 = stablehlo.constant dense<[false, true]> : tensor<2xi1>
  %2 = stablehlo.xor %0, %1 : tensor<2xi1>
  check.expect_eq_const %2, dense<[false, true]> : tensor<2xi1>
  func.return
}

// -----

func.func @xor_op_test_i1_splat_true() {
  %0 = stablehlo.constant dense<true> : tensor<2xi1>
  %1 = stablehlo.constant dense<[false, true]> : tensor<2xi1>
  %2 = stablehlo.xor %0, %1 : tensor<2xi1>
  check.expect_eq_const %2, dense<[true, false]> : tensor<2xi1>
  func.return
}
