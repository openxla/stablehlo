// RUN: stablehlo-translate --interpret -split-input-file %s


module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[[1.000000e+00], [2.000000e+00], [5.000000e+00], [6.000000e+00]], [[3.000000e+00], [4.000000e+00], [7.000000e+00], [8.000000e+00]], [[1.000000e+01], [1.100000e+01], [1.400000e+01], [1.500000e+01]], [[1.200000e+01], [1.300000e+01], [1.600000e+01], [1.700000e+01]]]]> : tensor<1x4x4x1xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<3x3x1x1xf32>
    %cst_1 = stablehlo.constant dense<2.47494364> : tensor<1x2x2x1xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x3x1x1xf32>) -> tensor<3x3x1x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [4, 4], lhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x4x1x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<3x3x1x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>) -> tensor<1x2x2x1x!quant.uniform<i32:f32, 1.5350925350904778E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<1x2x2x1x!quant.uniform<i32:f32, 1.5350925350904778E-5>>) -> tensor<1x2x2x1x!quant.uniform<i8:f32, 0.0097056613248937273:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<1x2x2x1x!quant.uniform<i8:f32, 0.0097056613248937273:-128>>) -> tensor<1x2x2x1xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
