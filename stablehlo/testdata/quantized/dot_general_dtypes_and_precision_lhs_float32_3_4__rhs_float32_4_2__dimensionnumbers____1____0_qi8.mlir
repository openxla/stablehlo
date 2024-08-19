module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[-0.648610889, -4.839990e-01, 3.39964437, 0.349830806], [-4.12569952, -6.90287971, -0.153646722, 5.38082075], [-2.10003686, -0.173380762, 2.26172876, 1.9670006]]> : tensor<3x4xf32>
    %cst_0 = stablehlo.constant dense<[[1.9392488, -1.40549958], [-3.80043983, 3.44176579], [-3.12474394, 0.0999774113], [-2.64203429, -2.605490e+00]]> : tensor<4x2xf32>
    %cst_1 = stablehlo.constant dense<[[0.000000e+00, 0.0983186662], [0.000000e+00, 0.000000e+00], [0.000000e+00, 0.0983186662]]> : tensor<3x2xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<4x2xf32>) -> tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0] : (tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>, tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>) -> tensor<3x2xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
