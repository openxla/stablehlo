// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[5.81311798, 2.08485532, 0.151162371], [-1.21007407, -1.59476554, 0.846119463], [-0.83784312, -0.416278511, 1.24929118], [3.46354723, 2.21915126, 3.81866336]]> : tensor<4x3xf32>
    %cst_0 = stablehlo.constant dense<[[-2.10215521, -1.803730e+00, -7.83739519, 4.36787844, 1.4788357, 3.10357666], [-4.46420813, 0.879630148, -2.18081808, -1.95115197, -3.56435633, -0.671983778], [-2.76886797, -0.212248296, 2.77085519, -1.21441388, -3.28464937, -4.60568237]]> : tensor<3x6xf32>
    %cst_1 = stablehlo.constant dense<[[0.000000e+00, 0.880177557, 0.148240432, 1.00062287, 1.00062287, 1.00062287], [0.000000e+00, 0.000000e+00, 0.843117476, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.00062287, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.880177557, 1.00062287, 1.00062287, 1.00062287, 1.00062287]]> : tensor<4x6xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x6xf32>) -> tensor<3x6x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0] : (tensor<4x3x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>, tensor<3x6x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>) -> tensor<4x6x!quant.uniform<i32:f32, 1.5342546266746989E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<4x6x!quant.uniform<i32:f32, 1.5342546266746989E-5>>) -> tensor<4x6x!quant.uniform<i8:f32, 0.0092650273266960594:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<4x6x!quant.uniform<i8:f32, 0.0092650273266960594:-128>>) -> tensor<4x6xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
