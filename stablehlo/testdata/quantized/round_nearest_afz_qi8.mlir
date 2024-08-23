// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[-2.500000e+00, 4.000000e-01, 5.000000e-01, 6.000000e-01, 2.500000e+00]> : tensor<5xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<5xf32>
    %0 = stablehlo.uniform_quantize %cst : (tensor<5xf32>) -> tensor<5x!quant.uniform<i8:f32, 0.0038905945478701124:-1>>
    %1 = stablehlo.round_nearest_afz %0 : (tensor<5x!quant.uniform<i8:f32, 0.0038905945478701124:-1>>) -> tensor<5x!quant.uniform<i8:f32, 7.843137254901961E-9>>
    %2 = stablehlo.uniform_dequantize %1 : (tensor<5x!quant.uniform<i8:f32, 7.843137254901961E-9>>) -> tensor<5xf32>
    %3 = stablehlo.custom_call @check.eq(%cst_0, %2) : (tensor<5xf32>, tensor<5xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
}
