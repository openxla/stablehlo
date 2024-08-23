// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[3.21390557, 2.41580057, -0.537137687, -1.02739561], [1.35577726, -2.48765302, 3.99296689, 3.09424305], [-2.41700459, -1.63692343, -6.27982473, 2.19841671]]]> : tensor<1x3x4xf32>
    %cst_0 = stablehlo.constant dense<[[[-0.227901727, -0.527938426, 2.04744601], [0.46251452, -2.02832699, -5.24830675], [-1.4312098, -5.60030842, 6.4256258], [-0.754353106, -7.84103918, -1.26745594]]]> : tensor<1x4x3xf32>
    %cst_1 = stablehlo.constant dense<-0.372528762> : tensor<1xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [0], contracting_dims = [2, 1] x [1, 2] : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
