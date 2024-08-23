// RUN: stablehlo-translate --interpret -split-input-file %s

module {
  func.func @tanh_op_test_bf16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>
    %0 = stablehlo.tanh %cst : tensor<11xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 7.617180e-01, 1.245120e-01, 9.960930e-02, 9.960930e-01, 1.000000e+00, -1.000000e+00, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    return
  }
}

// -----

module {
  func.func @tanh_op_test_f16() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>
    %0 = stablehlo.tanh %cst : tensor<11xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 7.617180e-01, 1.243290e-01, 9.967040e-02, 9.960930e-01, 1.000000e+00, -1.000000e+00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    return
  }
}

// -----

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<11xf32> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 0.761130392, 0.12536265, 0.101484053, 0.761130392, 0.761130392, 0.000000e+00, 0.462647885, 0.000000e+00, 0.000000e+00]> : tensor<11xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.tanh %0 : (tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<11x!quant.uniform<i8:f32, 0.0029848250688291065:-128>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<11x!quant.uniform<i8:f32, 0.0029848250688291065:-128>>) -> tensor<11xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<11xf32>, tensor<11xf32>) -> tensor<i1>
    return %2 : tensor<11xf32>
  }
}

// -----

module {
  func.func @tanh_op_test_f64() {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>
    %0 = stablehlo.tanh %cst : tensor<11xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 0.76159415595576485, 0.12435300177159619, 0.099667994624955819, 0.99627207622074998, 1.000000e+00, -1.000000e+00, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    return
  }
}

// -----

module {
  func.func @tanh_op_test_c32() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>
    %0 = stablehlo.tanh %cst : tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(0.967786788,-0.0926378369), (1.00166273,7.52857188E-4)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    return
  }
}

// -----

module {
  func.func @tanh_op_test_c64() {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>
    %0 = stablehlo.tanh %cst : tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(0.96778680215277412,-0.092637836268419898), (1.0016627850956348,7.5285721538218659E-4)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    return
  }
}
