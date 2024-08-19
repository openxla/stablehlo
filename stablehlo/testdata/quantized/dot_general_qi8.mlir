// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], [[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]> : tensor<2x2x2xf32>
    %cst_0 = stablehlo.constant dense<[[[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]], [[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]]> : tensor<2x2x2xf32>
    %cst_1 = stablehlo.constant dense<0.996095359> : tensor<2x2x2xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<2x2x2xf32>) -> tensor<2x2x2x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<2x2x2xf32>) -> tensor<2x2x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x2x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<2x2x2x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<2x2x2x!quant.uniform<i32:f32, 1.5329303653522721E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<2x2x2x!quant.uniform<i32:f32, 1.5329303653522721E-5>>) -> tensor<2x2x2x!quant.uniform<i8:f32, 0.0058251189250572051:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<2x2x2x!quant.uniform<i8:f32, 0.0058251189250572051:-128>>) -> tensor<2x2x2xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}