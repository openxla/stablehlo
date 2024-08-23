// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<1xf32> {
    %cst = stablehlo.constant dense<0x7F800000> : tensor<1xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<1xf32>) -> tensor<1x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>
    %1 = stablehlo.divide %0, %0 : (tensor<1x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>, tensor<1x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<1x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>) -> tensor<1xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %2 : tensor<1xf32>
  }
}
