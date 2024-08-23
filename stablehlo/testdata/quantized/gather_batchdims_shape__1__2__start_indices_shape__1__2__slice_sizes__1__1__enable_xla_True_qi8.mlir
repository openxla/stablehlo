// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<[[0, 1]]> : tensor<1x2xi32>
    %cst = stablehlo.constant dense<[[-2.72349977, -0.208018199]]> : tensor<1x2xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<1x2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>, tensor<1x2xi32>) -> tensor<1x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<1x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.0038181253508025523:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<1x!quant.uniform<i8:f32, 0.0038181253508025523:-128>>) -> tensor<1xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
