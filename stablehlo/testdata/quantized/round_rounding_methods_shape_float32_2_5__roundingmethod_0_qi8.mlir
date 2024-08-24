// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[5.000000e-01, 1.200000e+00, 1.500000e+00, 1.700000e+00, 2.500000e+00], [-5.000000e-01, -1.200000e+00, -1.500000e+00, -1.700000e+00, -2.500000e+00]]> : tensor<2x5xf32>
    %cst_0 = stablehlo.constant dense<[[0.996078491, 0.996078491, 0.996078491, 0.996078491, 0.996078491], [-0.996078491, -0.996078491, -0.996078491, -0.996078491, -0.996078491]]> : tensor<2x5xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<2x5xf32>) -> tensor<2x5x!quant.uniform<i8:f32, 0.0078306291617599184>>
    %1 = stablehlo.round_nearest_afz %0 : (tensor<2x5x!quant.uniform<i8:f32, 0.0078306291617599184>>) -> tensor<2x5x!quant.uniform<i8:f32, 0.0078431372549019607:-1>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<2x5x!quant.uniform<i8:f32, 0.0078431372549019607:-1>>) -> tensor<2x5xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
}
