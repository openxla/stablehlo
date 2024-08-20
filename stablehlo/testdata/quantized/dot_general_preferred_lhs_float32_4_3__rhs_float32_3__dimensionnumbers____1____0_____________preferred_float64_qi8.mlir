// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[1.51487315, 1.84060729, 1.371032], [-2.20898318, -5.76905251, -2.57896972], [-1.58193374, -2.16250372, -4.66199541], [0.347719163, 8.04571342, -9.58430767]]> : tensor<4x3xf32>
    %cst_0 = stablehlo.constant dense<[-0.860122799, -1.25689316, -3.9581697]> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 0.0039101013950273104:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0] : (tensor<4x3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<3x!quant.uniform<i8:f32, 0.0039101013950273104:-128>>) -> tensor<4x!quant.uniform<i32:f32, 1.5316809876069596E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<4x!quant.uniform<i32:f32, 1.5316809876069596E-5>>) -> tensor<4x!quant.uniform<i8:f32, 0.0077393209233003503:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<4x!quant.uniform<i8:f32, 0.0077393209233003503:-128>>) -> tensor<4xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
