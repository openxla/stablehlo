// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[2.9270215, 7.86154318, -5.63383484, 1.18890381], [1.66500914, -0.686581432, -1.0598495, 3.66114569], [-2.12638235, -5.93207598, 1.81490195, 0.333228439], [-0.129492328, 5.85269737, 1.17887712, -3.05277419]]> : tensor<4x4xf32>
    %cst_0 = stablehlo.constant dense<[0.148809016, 4.21798277, -8.70141696, -2.01860809]> : tensor<4xf32>
    %cst_1 = stablehlo.constant dense<[1.1417222, 0.146374643, 0.000000e+00, 0.995347559]> : tensor<4xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<4x4xf32>) -> tensor<4x4x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0] : (tensor<4x4x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<4x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>) -> tensor<4x!quant.uniform<i32:f32, 1.5304182416571165E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<4x!quant.uniform<i32:f32, 1.5304182416571165E-5>>) -> tensor<4x!quant.uniform<i8:f32, 0.0097583097570082718:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<4x!quant.uniform<i8:f32, 0.0097583097570082718:-128>>) -> tensor<4xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
