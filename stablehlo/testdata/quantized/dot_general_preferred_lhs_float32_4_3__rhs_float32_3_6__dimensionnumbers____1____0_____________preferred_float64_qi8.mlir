// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[-3.74693418, 2.34398603, 2.1041081], [-8.42414665, -0.577481806, -2.22710657], [4.59838724, -2.86711073, 0.0989064946], [0.679040133, 2.55961132, -6.06046581]]> : tensor<4x3xf32>
    %cst_0 = stablehlo.constant dense<[[-5.99892235, 0.891323149, 1.7891463, 4.15803719, 4.00103331, 1.5682838], [3.05114746, -5.205091, 1.55728626, 1.05174971, -1.59619689, -2.16938281], [1.82395577, -0.112828031, 2.08077788, 2.36720943, -4.93377209, -2.33540392]]> : tensor<3x6xf32>
    %cst_1 = stablehlo.constant dense<[[1.99198079, 0.000000e+00, 1.99198079, 1.99198079, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.1019153, 0.889442563, 1.09327316, 1.09327316, 1.00062287, 1.00062287], [1.00062287, 0.602226734, 1.67696989, 1.67696989, 0.676346957, 0.676346957]]> : tensor<4x6xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x6xf32>) -> tensor<3x6x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0] : (tensor<4x3x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>, tensor<3x6x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>) -> tensor<4x6x!quant.uniform<i32:f32, 1.5342546266746989E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<4x6x!quant.uniform<i32:f32, 1.5342546266746989E-5>>) -> tensor<4x6x!quant.uniform<i8:f32, 0.0092650273266960594:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<4x6x!quant.uniform<i8:f32, 0.0092650273266960594:-128>>) -> tensor<4x6xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
