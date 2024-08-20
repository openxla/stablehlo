// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<[[[0, 0], [1, 0], [2, 1]], [[0, 1], [1, 1], [0, 9]]]> : tensor<2x3x2xi32>
    %cst = stablehlo.constant dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]], [[9.000000e+00, 1.000000e+01], [1.100000e+01, 1.200000e+01], [1.300000e+01, 1.400000e+01], [1.500000e+01, 1.600000e+01]], [[1.700000e+01, 1.800000e+01], [1.900000e+01, 2.000000e+01], [2.100000e+01, 2.200000e+01], [2.300000e+01, 2.400000e+01]]]> : tensor<3x4x2xf32>
    %cst_0 = stablehlo.constant dense<0.998896479> : tensor<2x3x2x2xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<3x4x2xf32>) -> tensor<3x4x2x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 2, 2>}> : (tensor<3x4x2x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>, tensor<2x3x2xi32>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<2x3x2x2x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<2x3x2x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<2x3x2x2xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<2x3x2x2xf32>, tensor<2x3x2x2xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
