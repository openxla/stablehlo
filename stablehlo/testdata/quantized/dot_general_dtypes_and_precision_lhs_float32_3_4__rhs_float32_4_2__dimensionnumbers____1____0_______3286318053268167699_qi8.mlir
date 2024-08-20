// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[-2.98286057, 1.45473039, 4.37298536, 4.06851482], [-3.93359685, 5.34374475, -7.59047747, -1.32499611], [-4.2501359, 4.22664261, 3.23046494, 7.55214739]]> : tensor<3x4xf32>
    %cst_0 = stablehlo.constant dense<[[-2.42577767, 1.8425138], [-2.80951762, -6.45209217], [0.841208279, -3.3215766], [0.892308592, -3.71017742]]> : tensor<4x2xf32>
    %cst_1 = stablehlo.constant dense<[[1.73040843, 0.000000e+00], [0.000000e+00, 0.000000e+00], [1.73040843, 0.000000e+00]]> : tensor<3x2xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<4x2xf32>) -> tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>, tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>) -> tensor<3x2xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
