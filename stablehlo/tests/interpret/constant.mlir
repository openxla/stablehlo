// RUN: stablehlo-translate --interpret -split-input-file %s

func.func @constant_op_test_si2() {
  %0 = stablehlo.constant dense<[-2, -1, 0, 1]> : tensor<4xi2>
  check.expect_eq_const %0, dense<[-2, -1, 0, 1]> : tensor<4xi2>
  func.return
}

// -----

func.func @constant_op_test_ui2() {
  %0 = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xui2>
  check.expect_eq_const %0, dense<[0, 1, 2, 3]> : tensor<4xui2>
  func.return
}

// -----

func.func @constant_op_test_si4() {
  %0 = stablehlo.constant dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>
  check.expect_eq_const %0, dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>
  func.return
}

// -----

func.func @constant_op_test_ui4() {
  %0 = stablehlo.constant dense<[0, 8, 15]> : tensor<3xui4>
  check.expect_eq_const %0, dense<[0, 8, 15]> : tensor<3xui4>
  func.return
}

// -----

func.func @constant_op_test_si8() {
  %0 = stablehlo.constant dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>
  check.expect_eq_const %0, dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>
  func.return
}

// -----

func.func @constant_op_test_ui8() {
  %0 = stablehlo.constant dense<[0, 16, 255]> : tensor<3xui8>
  check.expect_eq_const %0, dense<[0, 16, 255]> : tensor<3xui8>
  func.return
}

// -----

func.func @constant_op_test_si16() {
  %0 = stablehlo.constant dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>
  check.expect_eq_const %0, dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>
  func.return
}

// -----

func.func @constant_op_test_ui16() {
  %0 = stablehlo.constant dense<[0, 256, 65535]> : tensor<3xui16>
  check.expect_eq_const %0, dense<[0, 256, 65535]> : tensor<3xui16>
  func.return
}

// -----

func.func @constant_op_test_si32() {
  %0 = stablehlo.constant dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>
  check.expect_eq_const %0, dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>
  func.return
}

// -----

func.func @constant_op_test_ui32() {
  %0 = stablehlo.constant dense<[0, 65536, 4294967295]> : tensor<3xui32>
  check.expect_eq_const %0, dense<[0, 65536, 4294967295]> : tensor<3xui32>
  func.return
}

// -----

func.func @constant_op_test_si64() {
  %0 = stablehlo.constant dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>
  check.expect_eq_const %0, dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>
  func.return
}

// -----

func.func @constant_op_test_ui64() {
  %0 = stablehlo.constant dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>
  check.expect_eq_const %0, dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>
  func.return
}

// -----

func.func @constant_op_test_f4_e2m1fn() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x07, 0x0F, 0x01, 0x09]> : tensor<10xf4E2M1FN>
  check.expect_almost_eq_const %0, dense<[0.0, -0.0, 1.0, 0.0, 0.0, 3.0, 6.0, -6.0, 0.5, -0.5]> : tensor<10xf4E2M1FN>
  func.return
}

// -----

func.func @constant_op_test_f6_e2m3fn() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x1F, 0x3F, 0x01, 0x21]> : tensor<10xf6E2M3FN>
  check.expect_almost_eq_const %0, dense<[0.0, -0.0, 1.0, 0.125, 0.125, 3.25, 7.5, -7.5, 0.125, -0.125]> : tensor<10xf6E2M3FN>
  func.return
}

// -----

func.func @constant_op_test_f6_e3m2fn() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x1F, 0x3F, 0x01, 0x21]> : tensor<10xf6E3M2FN>
  check.expect_almost_eq_const %0, dense<[0.0, -0.0, 1.0, 0.125, 0.125, 3.0, 28.0, -28.0, 0.0625, -0.0625]> : tensor<10xf6E3M2FN>
  func.return
}

// -----

func.func @constant_op_test_f8_e3m4() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E3M4>
  check.expect_almost_eq_const %0, dense<[0.0, -0.0, 1.0, 0.125, 0.09375, 3.125, 0x7F, 0xFF, 0.015625, -0.015625]> : tensor<10xf8E3M4>
  func.return
}

// -----

func.func @constant_op_test_f8_e4m3b11_fnuz() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E4M3B11FNUZ>
  check.expect_almost_eq_const %0, dense<[0.0, 0.0, 1.0, 0.125, 0.101563, 3.25, 30.0, -30.0, 0.00012207, -0.00012207]> : tensor<10xf8E4M3B11FNUZ>
  func.return
}

// -----

func.func @constant_op_test_f8_e4m3() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E4M3>
  check.expect_almost_eq_const %0, dense<[0.0, -0.0, 1.0, 0.125, 0.1015630, 3.25, 0x7F, 0xFF, 0.001953130, -0.001953130]> : tensor<10xf8E4M3>
  func.return
}

// -----

func.func @constant_op_test_f8_e4m3_fn() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E4M3FN>
  check.expect_almost_eq_const %0, dense<[0.0, -0.0, 1.0, 0.125, 0.1015630, 3.25, 0x7F, 0xFF, 0.001953130, -0.001953130]> : tensor<10xf8E4M3FN>
  func.return
}

// -----

func.func @constant_op_test_f8_e4m3_fnuz() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E4M3FNUZ>
  check.expect_almost_eq_const %0, dense<[0.0, 0.0, 1.0, 0.125, 0.101563, 3.25, 240.0, -240.0, 0.000976562, -0.0009765620]> : tensor<10xf8E4M3FNUZ>
  func.return
}

// -----

func.func @constant_op_test_f8_e5m2() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E5M2>
  check.expect_almost_eq_const %0, dense<[0.0, -0.0, 1.0, 0.125, 0.09375, 3.0, 0x7F, 0xFF, 0.0000152588, -0.0000152588]> : tensor<10xf8E5M2>
  func.return
}

// -----

func.func @constant_op_test_f8_e5m2_fnuz() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E5M2FNUZ>
  check.expect_almost_eq_const %0, dense<[0.0, 0.0, 1.0, 0.125, 0.09375, 3.0, 57344.0, -57344.0, 0.00000762939, -0.00000762939]> : tensor<10xf8E5M2FNUZ>
  func.return
}

// -----
func.func @constant_op_test_f8_e5m2_fnuz() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415, 0x7F, 0xFF, 0x01, 0x81]> : tensor<10xf8E5M2FNUZ>
  check.expect_eq_const %0, dense<[0.0, 0.0, 1.0, 0.125, 0.1, 3.0, 57344.0, -57344.0, 7.62939e-06, -7.62939e-06]> : tensor<10xf8E5M2FNUZ>
  func.return
}

// -----

func.func @constant_op_test_f8_e8m0fnu() {
  %0 = stablehlo.constant dense<[0.0, 1.0, 0.125, 0.1, 3.1415, 0x00, 0x80, 0xFF]> : tensor<8xf8E8M0FNU>
  check.expect_almost_eq_const %0, dense<[0.0, 1.0, 0.125, 0.125, 4.0, 1.175490e-38, 2.0, 0xFF]> : tensor<8xf8E8M0FNU>
  func.return
}

// -----

func.func @constant_op_test_bf16() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001]> : tensor<11xbf16>
  check.expect_almost_eq_const %0, dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>
  func.return
}

// -----

func.func @constant_op_test_f16() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.140630, 0x7C00, 0xFC00, 0x7FFF, 0x0001, 0x8001]> : tensor<11xf16>
  check.expect_almost_eq_const %0, dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>
  func.return
}

// -----

func.func @constant_op_test_f32() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 0x00000001, 0x80000001]> : tensor<11xf32>
  check.expect_almost_eq_const %0, dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>
  func.return
}

// -----

func.func @constant_op_test_f64() {
  %0 = stablehlo.constant dense<[0.0, -0.0, 1.0, 0.125, 0.1, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0x8000000000000001]> : tensor<11xf64>
  check.expect_almost_eq_const %0, dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>
  func.return
}

// -----

func.func @constant_op_test_c64() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %0, dense<[(1.500000e+00, 2.500000e+00), (3.500000e+00, 4.500000e+00)]> : tensor<2xcomplex<f32>>
  func.return
}

// -----

func.func @constant_op_test_c128() {
  %0 = stablehlo.constant dense<[(1.5, 2.5), (3.5, 4.5)]> : tensor<2xcomplex<f64>>
  check.expect_almost_eq_const %0, dense<[(1.500000e+00, 2.500000e+00), (3.500000e+00, 4.500000e+00)]> : tensor<2xcomplex<f64>>
  func.return
}
