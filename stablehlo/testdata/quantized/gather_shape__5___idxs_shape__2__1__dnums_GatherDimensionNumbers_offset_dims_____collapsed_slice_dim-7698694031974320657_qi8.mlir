// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %cst = stablehlo.constant dense<[-4.45134878, -1.72604203, 5.85744715, -1.34194362, 0.103943698]> : tensor<5xf32>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, 0.984438836]> : tensor<2xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<5xf32>) -> tensor<5x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<5x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>, tensor<2x1xi32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<2x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>) -> tensor<2x!quant.uniform<i8:f32, 0.0038605444571551155:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<2x!quant.uniform<i8:f32, 0.0038605444571551155:-128>>) -> tensor<2xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
