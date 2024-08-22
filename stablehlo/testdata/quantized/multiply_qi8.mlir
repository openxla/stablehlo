// RUN: stablehlo-translate --interpret -split-input-file %s

module {
  func.func @mul_op_test_si8() {
    %c = stablehlo.constant dense<[0, 1, 8, -9, 0]> : tensor<5xi8>
    %c_0 = stablehlo.constant dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>
    %0 = stablehlo.multiply %c, %c_0 : tensor<5xi8>
    "check.expect_eq_const"(%0) <{value = dense<[0, -1, 64, 81, 0]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_ui8() {
    %c = stablehlo.constant dense<[0, 16, 16]> : tensor<3xui8>
    %c_0 = stablehlo.constant dense<[255, 16, 17]> : tensor<3xui8>
    %0 = stablehlo.multiply %c, %c_0 : tensor<3xui8>
    "check.expect_eq_const"(%0) <{value = dense<[0, 0, 16]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_si16() {
    %c = stablehlo.constant dense<[0, 1, 128, -129, 0]> : tensor<5xi16>
    %c_0 = stablehlo.constant dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>
    %0 = stablehlo.multiply %c, %c_0 : tensor<5xi16>
    "check.expect_eq_const"(%0) <{value = dense<[0, -1, 16384, 16641, 0]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_ui16() {
    %c = stablehlo.constant dense<[0, 256]> : tensor<2xui16>
    %c_0 = stablehlo.constant dense<[65535, 256]> : tensor<2xui16>
    %0 = stablehlo.multiply %c, %c_0 : tensor<2xui16>
    "check.expect_eq_const"(%0) <{value = dense<0> : tensor<2xui16>}> : (tensor<2xui16>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_si32() {
    %c = stablehlo.constant dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>
    %c_0 = stablehlo.constant dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>
    %0 = stablehlo.multiply %c, %c_0 : tensor<5xi32>
    "check.expect_eq_const"(%0) <{value = dense<[0, -1, 1073741824, 1073807361, 0]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_ui32() {
    %c = stablehlo.constant dense<[0, 65536]> : tensor<2xui32>
    %c_0 = stablehlo.constant dense<[4294967295, 65536]> : tensor<2xui32>
    %0 = stablehlo.multiply %c, %c_0 : tensor<2xui32>
    "check.expect_eq_const"(%0) <{value = dense<0> : tensor<2xui32>}> : (tensor<2xui32>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_si64() {
    %c = stablehlo.constant dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>
    %c_0 = stablehlo.constant dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>
    %0 = stablehlo.multiply %c, %c_0 : tensor<5xi64>
    "check.expect_eq_const"(%0) <{value = dense<[0, -1, 4611686018427387904, 4611686022722355201, 0]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_ui64() {
    %c = stablehlo.constant dense<[0, 4294967296]> : tensor<2xui64>
    %c_0 = stablehlo.constant dense<[18446744073709551615, 4294967296]> : tensor<2xui64>
    %0 = stablehlo.multiply %c, %c_0 : tensor<2xui64>
    "check.expect_eq_const"(%0) <{value = dense<0> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_i1() {
    %c = stablehlo.constant dense<[false, false, true, true]> : tensor<4xi1>
    %c_0 = stablehlo.constant dense<[false, true, false, true]> : tensor<4xi1>
    %0 = stablehlo.multiply %c, %c_0 : tensor<4xi1>
    "check.expect_eq_const"(%0) <{value = dense<[false, false, false, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_bf16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 2.658460e+36, 2.658460e+36, -2.658460e+36, 2.658460e+36, 9.183550e-41]> : tensor<11xbf16>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.007810e-01, 3.140630e+00, 0.000000e+00, 2.658460e+36, -2.658460e+36, -2.658460e+36, -9.183550e-41]> : tensor<11xbf16>
    %0 = stablehlo.multiply %cst, %cst_0 : tensor<11xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 3.015140e-02, 9.875000e+00, 0.000000e+00, 0x7F80, 0x7F80, 0xFF80, -0.000000e+00]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_f16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0x7C00, 0xFC00, 0x7C00, 5.960460e-08]> : tensor<11xf16>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000490e-01, 3.140630e+00, 0.000000e+00, 0x7C00, 0xFC00, 0xFC00, -5.960460e-08]> : tensor<11xf16>
    %0 = stablehlo.multiply %cst, %cst_0 : tensor<11xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 2.999880e-02, 9.867180e+00, 0x7E00, 0x7C00, 0x7C00, 0xFC00, -0.000000e+00]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    return
  }
}

// -----

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<11xf32> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0x7F800000, 0xFF800000, 0x7F800000, 1.401300e-45]> : tensor<11xf32>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.14159274, 0.000000e+00, 0x7F800000, 0xFF800000, 0xFF800000, -1.401300e-45]> : tensor<11xf32>
    %cst_1 = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 0.972242116, 0.0953178555, 0.030501714, 0.972242116, 0.000000e+00, 0.972242116, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<11xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.multiply %1, %0 : (tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<11x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>) -> tensor<11x!quant.uniform<i8:f32, 0.0038127141840317669:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<11x!quant.uniform<i8:f32, 0.0038127141840317669:-128>>) -> tensor<11xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_1, %3) : (tensor<11xf32>, tensor<11xf32>) -> tensor<i1>
    return %3 : tensor<11xf32>
  }
}

// -----

module {
  func.func @mul_op_test_f64() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000000, 4.940660e-324]> : tensor<11xf64>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.1415926535897931, 0.000000e+00, 0x7FF0000000000000, 0xFFF0000000000000, 0xFFF0000000000000, -4.940660e-324]> : tensor<11xf64>
    %0 = stablehlo.multiply %cst, %cst_0 : tensor<11xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 3.000000e-02, 9.869604401089358, 0x7FF8000000000000, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, -0.000000e+00]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_c64() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>
    %0 = stablehlo.multiply %cst, %cst : tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(-4.000000e+00,7.500000e+00), (2.600000e+01,8.250000e+01)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    return
  }
}

// -----

module {
  func.func @mul_op_test_c128() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>
    %0 = stablehlo.multiply %cst, %cst : tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(-4.000000e+00,7.500000e+00), (2.600000e+01,8.250000e+01)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    return
  }
}
