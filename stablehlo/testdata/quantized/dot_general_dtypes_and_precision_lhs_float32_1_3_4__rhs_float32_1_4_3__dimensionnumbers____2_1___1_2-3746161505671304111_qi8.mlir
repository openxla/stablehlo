// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[2.59234142, -2.35737705, 3.07461166, -5.77705336], [-3.64460349, 0.689637601, -0.0876942202, 1.62593222], [-4.65739489, 0.247004092, -1.08101177, -0.238710642]]]> : tensor<1x3x4xf32>
    %cst_0 = stablehlo.constant dense<[[[-3.64053726, 1.4618907, -0.867068588], [2.86438012, -2.70172548, -0.4580172], [0.140140817, -0.666462898, -7.101100e-01], [1.02142382, 0.236523077, 0.760420739]]]> : tensor<1x4x3xf32>
    %cst_1 = stablehlo.constant dense<-0.558793128> : tensor<1xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [0], contracting_dims = [2, 1] x [1, 2], precision = [HIGHEST, HIGHEST] : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
