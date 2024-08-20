// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[-3.91622734, -0.242899641, -1.58674753]> : tensor<3xf32>
    %cst_0 = stablehlo.constant dense<[[0.932529568, 1.69621098, 1.12284088, 4.28206635, 0.539385378, 2.11882901], [1.37038183, 3.67467952, -3.68408799, -0.532391131, -1.91454673, 0.2745637], [1.70064592, -0.347891033, -3.86588287, 0.385282725, -1.16977382, -2.22889447]]> : tensor<3x6xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x6xf32>) -> tensor<3x6x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [0] x [0] : (tensor<3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<3x6x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>) -> tensor<6x!quant.uniform<i32:f32, 1.5343835624351492E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<6x!quant.uniform<i32:f32, 1.5343835624351492E-5>>) -> tensor<6x!quant.uniform<i8:f32, 0.0072211433859432446:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<6x!quant.uniform<i8:f32, 0.0072211433859432446:-128>>) -> tensor<6xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<6xf32>, tensor<6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
