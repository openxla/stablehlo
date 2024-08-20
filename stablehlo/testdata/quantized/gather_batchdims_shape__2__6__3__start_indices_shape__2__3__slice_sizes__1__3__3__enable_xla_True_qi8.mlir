// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<[[0, 1, 0], [1, 2, 0]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[[3.82548904, 0.791181862, -2.22872925], [1.47356987, 3.81562257, -4.14531422], [-2.14515972, 1.42112124, 4.571450e+00], [-1.68962431, 3.14189243, 5.90857506], [3.88763309, 3.85987115, 0.356856197], [0.954816877, 1.0329355, -0.830992698]], [[-2.23060846, 0.0469221957, -0.450263053], [-6.04691744, 6.02806186, -2.51375771], [0.53378284, 3.34858298, 1.84060633], [3.92621756, -1.48187923, 3.34925771], [-2.8641305, 0.439090401, 4.06969261], [-7.4090166, 3.41720462, -4.32454443]]]> : tensor<2x6x3xf32>
    %cst_0 = stablehlo.constant dense<[[[0.998835206, 0.998835206, 0.000000e+00], [0.000000e+00, 0.998835206, 0.998835206], [0.000000e+00, 0.998835206, 0.998835206]], [[0.532712102, 0.998835206, 0.998835206], [0.998835206, 0.000000e+00, 0.998835206], [0.000000e+00, 0.438704073, 0.998835206]]]> : tensor<2x3x3xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<2x6x3xf32>) -> tensor<2x6x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3, 3>}> : (tensor<2x6x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>, tensor<2x3xi32>) -> tensor<2x3x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<2x3x3x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<2x3x3x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<2x3x3x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>) -> tensor<2x3x3xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<2x3x3xf32>, tensor<2x3x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
