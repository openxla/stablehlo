module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[-3.35913014, -1.78181207, 1.45402408]> : tensor<3xf32>
    %cst_0 = stablehlo.constant dense<[3.0900166, -2.45028687, -4.84450102]> : tensor<3xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [0] x [0] : (tensor<3x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>, tensor<3x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>) -> tensor<!quant.uniform<i32:f32, 1.5252998015758598E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<!quant.uniform<i32:f32, 1.5252998015758598E-5>>) -> tensor<!quant.uniform<i8:f32, 0.0079389403848087094:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<!quant.uniform<i8:f32, 0.0079389403848087094:-128>>) -> tensor<f32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
