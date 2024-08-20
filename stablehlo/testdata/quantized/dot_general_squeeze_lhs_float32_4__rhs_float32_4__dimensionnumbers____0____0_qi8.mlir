// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[-1.73818827, 6.32115507, 2.81545162, -1.37914991]> : tensor<4xf32>
    %cst_0 = stablehlo.constant dense<[-4.02553225, -2.70646834, 3.14252234, 1.59961236]> : tensor<4xf32>
    %cst_1 = stablehlo.constant dense<0.992584764> : tensor<f32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 0.003902135643304563:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [0] x [0] : (tensor<4x!quant.uniform<i8:f32, 0.003902135643304563:-128>>, tensor<4x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>) -> tensor<!quant.uniform<i32:f32, 1.5245167256298701E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<!quant.uniform<i32:f32, 1.5245167256298701E-5>>) -> tensor<!quant.uniform<i8:f32, 0.0081359405143588189:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<!quant.uniform<i8:f32, 0.0081359405143588189:-128>>) -> tensor<f32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
