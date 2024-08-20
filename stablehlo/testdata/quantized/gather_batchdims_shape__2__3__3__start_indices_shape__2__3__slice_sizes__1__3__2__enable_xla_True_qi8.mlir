// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<[[0, 1, 0], [1, 2, 1]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[[-2.31606197, -1.45022011, -1.72503948], [-4.47900438, -5.43648243, -4.72877312], [4.36842155, -1.49977052, -2.34371066]], [[-2.62882113, -3.40511084, 0.60867834], [-2.19209099, -0.954817473, -0.967517852], [-0.497551709, 6.707040e-01, -6.8893342]]]> : tensor<2x3x3xf32>
    %cst_0 = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00], [0.997889161, 0.000000e+00]], [[0.000000e+00, 0.606560051], [0.000000e+00, 0.000000e+00], [0.669172704, 0.000000e+00]]]> : tensor<2x3x2xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<2x3x3xf32>) -> tensor<2x3x3x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %1 = "stablehlo.gather"(%0, %c) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3, 2>}> : (tensor<2x3x3x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>, tensor<2x3xi32>) -> tensor<2x3x2x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %2 = stablehlo.uniform_quantize %1 : (tensor<2x3x2x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>) -> tensor<2x3x2x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<2x3x2x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<2x3x2xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_0, %3) : (tensor<2x3x2xf32>, tensor<2x3x2xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
