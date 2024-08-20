// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<[[0, 0], [1, 8], [2, 0]]> : tensor<3x2xi32>
    %cst = stablehlo.constant dense<[[-0.786750376, -0.429459691, -2.42140698, 0.0205181241, -0.394114822, -2.58621716, -1.07088399, 3.29197717, -3.44814229, -0.25225088], [1.27824605, -2.20641971, 1.13592541, 2.04215646, -1.61209357, 3.22753859, -1.28165495, 3.17407966, 2.02299929, 2.47564316], [0.905838906, 3.71254492, 1.97064459, 3.77753663, 1.49392521, 4.79311323, 3.70975041, -1.04468286, 3.31870532, 1.45112896]]> : tensor<3x10xf32>
    %cst_0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.0195749179, 0.000000e+00], [0.998320758, 0.000000e+00, 0.998320758, 0.998320758, 0.998320758], [0.904361188, 0.998320758, 0.998320758, 0.998320758, 0.998320758]]> : tensor<3x5xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<3x10xf32>) -> tensor<3x10x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 5>}> : (tensor<3x10x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>, tensor<3x2xi32>) -> tensor<3x5x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<3x5x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>) -> tensor<3x5x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<3x5x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>) -> tensor<3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
