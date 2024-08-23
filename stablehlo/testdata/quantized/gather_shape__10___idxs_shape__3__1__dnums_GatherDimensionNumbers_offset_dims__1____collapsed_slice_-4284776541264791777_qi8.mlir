// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<0> : tensor<3x1xi32>
    %cst = stablehlo.constant dense<[-0.317652494, -0.498273045, -1.63233531, -0.124743178, 2.18847871, 1.92351472, 1.37014866, -3.42049432, -2.30765843, 2.53218222]> : tensor<10xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<10xf32>) -> tensor<10x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 2>}> : (tensor<10x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<3x1xi32>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<3x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0038690987755270567:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<3x2x!quant.uniform<i8:f32, 0.0038690987755270567:-128>>) -> tensor<3x2xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
