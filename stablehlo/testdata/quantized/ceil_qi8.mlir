// RUN: stablehlo-translate --interpret -split-input-file %s

module {
  func.func @ceil_op_test_bf16() {
    %cst = stablehlo.constant dense<[0xFF80, -2.500000e+00, -9.183550e-41, -0.000000e+00, 0.000000e+00, 9.183550e-41, 2.500000e+00, 0x7F80, 0x7FC0]> : tensor<9xbf16>
    %0 = stablehlo.ceil %cst : tensor<9xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0xFF80, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 3.000000e+00, 0x7F80, 0x7FC0]> : tensor<9xbf16>}> : (tensor<9xbf16>) -> ()
    return
  }
}

// -----

module {
  func.func @ceil_op_test_f16() {
    %cst = stablehlo.constant dense<[0xFC00, -2.500000e+00, -5.960460e-08, -0.000000e+00, 0.000000e+00, 5.960460e-08, 2.500000e+00, 0x7C00, 0x7E00]> : tensor<9xf16>
    %0 = stablehlo.ceil %cst : tensor<9xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0xFC00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 3.000000e+00, 0x7C00, 0x7E00]> : tensor<9xf16>}> : (tensor<9xf16>) -> ()
    return
  }
}

// -----

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<9xf32> {
    %cst = stablehlo.constant dense<[0xFF800000, -2.500000e+00, -1.401300e-45, -0.000000e+00, 0.000000e+00, 1.401300e-45, 2.500000e+00, 0x7F800000, 0x7FC00000]> : tensor<9xf32>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<9xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<9xf32>) -> tensor<9x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
    %1 = stablehlo.ceil %0 : (tensor<9x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<9x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<9x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>) -> tensor<9xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<9xf32>, tensor<9xf32>) -> tensor<i1>
    return %2 : tensor<9xf32>
  }
}

// -----

module {
  func.func @ceil_op_test_f64() {
    %cst = stablehlo.constant dense<[0xFFF0000000000000, -2.500000e+00, -4.940660e-324, -0.000000e+00, 0.000000e+00, 4.940660e-324, 2.500000e+00, 0x7FF0000000000000, 0x7FF8000000000000]> : tensor<9xf64>
    %0 = stablehlo.ceil %cst : tensor<9xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0xFFF0000000000000, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 3.000000e+00, 0x7FF0000000000000, 0x7FF8000000000000]> : tensor<9xf64>}> : (tensor<9xf64>) -> ()
    return
  }
}
