// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 2]]> : tensor<4x2xi32>
    %cst = stablehlo.constant dense<[[-2.82557797, 2.39072633, 1.59782159, 5.14471102, -0.118122488, 1.23312056], [-1.81219053, -2.04905701, 2.10215306, -1.29667866, -0.0825303718, 1.88295043], [2.51706767, 0.0771943628, 2.18911791, -0.366536409, -2.39656186, 0.698230087], [2.96748114, 0.137859881, 1.44472873, -1.30095637, 1.24915195, -2.93037224]]> : tensor<4x6xf32>
    %cst_0 = stablehlo.constant dense<[[0.997595727, 0.997595727, 0.997595727], [0.997595727, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.696360946], [0.997595727, 0.000000e+00, 0.997595727]]> : tensor<4x3xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<4x6xf32>) -> tensor<4x6x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3>}> : (tensor<4x6x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<4x2xi32>) -> tensor<4x3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<4x3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<4x3x!quant.uniform<i8:f32, 0.0039121401076223335:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<4x3x!quant.uniform<i8:f32, 0.0039121401076223335:-128>>) -> tensor<4x3xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
