// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<2xf32> {
    %cst = stablehlo.constant dense<-1.000000e+00> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
    %cst_1 = stablehlo.constant dense<91.4060898> : tensor<2xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %2 = stablehlo.divide %1, %0 : (tensor<2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>, tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>) -> tensor<2x!quant.uniform<i8:f32, 0.71411007151884187:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<2x!quant.uniform<i8:f32, 0.71411007151884187:-128>>) -> tensor<2xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_1, %3) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %3 : tensor<2xf32>
  }
}
