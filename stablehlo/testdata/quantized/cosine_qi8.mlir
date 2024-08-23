// RUN: stablehlo-translate --interpret -split-input-file %s

module {
  func.func @cosine_op_test_bf16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>
    %0 = stablehlo.cosine %cst : tensor<11xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[1.000000e+00, 1.000000e+00, 5.390630e-01, 9.921870e-01, 9.960930e-01, -1.000000e+00, 0xFFC0, 0xFFC0, 0x7FFF, 1.000000e+00, 1.000000e+00]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    return
  }
}

// -----

module {
  func.func @cosine_op_test_f16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>
    %0 = stablehlo.cosine %cst : tensor<11xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[1.000000e+00, 1.000000e+00, 5.405270e-01, 9.921870e-01, 9.951170e-01, -1.000000e+00, 0xFE00, 0xFE00, 0x7FFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    return
  }
}

// -----

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<11xf32> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>
    %cst_0 = stablehlo.constant dense<[0.999999821, 0.999999821, 0.541176379, 0.992156684, 0.996078252, 0.541176379, 0.541176379, 0.999999821, 0.8784312, 0.999999821, 0.999999821]> : tensor<11xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.cosine %0 : (tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<11x!quant.uniform<i8:f32, 0.0039215681599635704:-128>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<11x!quant.uniform<i8:f32, 0.0039215681599635704:-128>>) -> tensor<11xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<11xf32>, tensor<11xf32>) -> tensor<i1>
    return %2 : tensor<11xf32>
  }
}

// -----

module {
  func.func @cosine_op_test_f64() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>
    %0 = stablehlo.cosine %cst : tensor<11xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[1.000000e+00, 1.000000e+00, 0.54030230586813977, 0.992197667229329, 0.99500416527802582, -1.000000e+00, 0xFFF8000000000000, 0xFFF8000000000000, 0x7FFFFFFFFFFFFFFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    return
  }
}

// -----

module {
  func.func @cosine_op_test_c64() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>
    %0 = stablehlo.cosine %cst : tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(4.337810e-01,-6.03504848), (-42.1537743,15.7863016)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    return
  }
}

// -----

module {
  func.func @cosine_op_test_c128() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>
    %0 = stablehlo.cosine %cst : tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(0.43378099760770306,-6.0350486377665726), (-42.153773835602316,15.786301507647636)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    return
  }
}
