// RUN: stablehlo-translate --interpret -split-input-file %s

module {
  func.func @negate_op_test_si4() {
    %c = stablehlo.constant dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>
    %0 = stablehlo.negate %c : tensor<5xi4>
    "check.expect_eq_const"(%0) <{value = dense<[-8, 1, 0, -1, -7]> : tensor<5xi4>}> : (tensor<5xi4>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_ui4() {
    %c = stablehlo.constant dense<[0, 8, 15]> : tensor<3xui4>
    %0 = stablehlo.negate %c : tensor<3xui4>
    "check.expect_eq_const"(%0) <{value = dense<[0, 8, 1]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_si8() {
    %c = stablehlo.constant dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>
    %0 = stablehlo.negate %c : tensor<5xi8>
    "check.expect_eq_const"(%0) <{value = dense<[-128, 9, 0, -8, -127]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_ui8() {
    %c = stablehlo.constant dense<[0, 16, 255]> : tensor<3xui8>
    %0 = stablehlo.negate %c : tensor<3xui8>
    "check.expect_eq_const"(%0) <{value = dense<[0, 240, 1]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_si16() {
    %c = stablehlo.constant dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>
    %0 = stablehlo.negate %c : tensor<5xi16>
    "check.expect_eq_const"(%0) <{value = dense<[-32768, 129, 0, -128, -32767]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_ui16() {
    %c = stablehlo.constant dense<[0, 256, 65535]> : tensor<3xui16>
    %0 = stablehlo.negate %c : tensor<3xui16>
    "check.expect_eq_const"(%0) <{value = dense<[0, 65280, 1]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_si32() {
    %c = stablehlo.constant dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>
    %0 = stablehlo.negate %c : tensor<5xi32>
    "check.expect_eq_const"(%0) <{value = dense<[-2147483648, 65537, 0, -65536, -2147483647]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_ui32() {
    %c = stablehlo.constant dense<[0, 65536, 4294967295]> : tensor<3xui32>
    %0 = stablehlo.negate %c : tensor<3xui32>
    "check.expect_eq_const"(%0) <{value = dense<[0, 4294901760, 1]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_si64() {
    %c = stablehlo.constant dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>
    %0 = stablehlo.negate %c : tensor<5xi64>
    "check.expect_eq_const"(%0) <{value = dense<[-9223372036854775808, 2147483649, 0, -2147483648, -9223372036854775807]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_ui64() {
    %c = stablehlo.constant dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>
    %0 = stablehlo.negate %c : tensor<3xui64>
    "check.expect_eq_const"(%0) <{value = dense<[0, 18446744069414584320, 1]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_bf16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>
    %0 = stablehlo.negate %cst : tensor<11xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[-0.000000e+00, 0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000980e-01, -3.140630e+00, 0xFF80, 0x7F80, 0xFFFF, -9.183550e-41, 9.183550e-41]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_f16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>
    %0 = stablehlo.negate %cst : tensor<11xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[-0.000000e+00, 0.000000e+00, -1.000000e+00, -1.250000e-01, -9.997550e-02, -3.140630e+00, 0xFC00, 0x7C00, 0xFFFF, -5.960460e-08, 5.960460e-08]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    return
  }
}

// -----

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<11xf32> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, -0.998896479, -0.125351712, -0.101848267, -0.998896479, -0.998896479, 0.000000e+00, -0.501406848, 0.000000e+00, 0.000000e+00]> : tensor<11xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.negate %0 : (tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:127>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:127>>) -> tensor<11xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<11xf32>, tensor<11xf32>) -> tensor<i1>
    return %2 : tensor<11xf32>
  }
}

// -----

module {
  func.func @negate_op_test_f64() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>
    %0 = stablehlo.negate %cst : tensor<11xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[-0.000000e+00, 0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000000e-01, -3.1415926535897931, 0xFFF0000000000000, 0x7FF0000000000000, 0xFFFFFFFFFFFFFFFF, -4.940660e-324, 4.940660e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_c64() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>
    %0 = stablehlo.negate %cst : tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(-1.500000e+00,-2.500000e+00), (-3.500000e+00,-4.500000e+00)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    return
  }
}

// -----

module {
  func.func @negate_op_test_c128() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>
    %0 = stablehlo.negate %cst : tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(-1.500000e+00,-2.500000e+00), (-3.500000e+00,-4.500000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    return
  }
}
