// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[-3.304690e+00, 3.608400e-01, 1.429690e+00]> : tensor<3xf16>
    %cst_0 = stablehlo.constant dense<[1.038090e+00, 6.635740e-01, 1.024410e+00]> : tensor<3xf16>
    %cst_1 = stablehlo.constant dense<1.23062062> : tensor<f32>
    %0 = stablehlo.convert %cst_0 : (tensor<3xf16>) -> tensor<3xf32>
    %1 = stablehlo.uniform_quantize %0 : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
    %2 = stablehlo.convert %cst : (tensor<3xf16>) -> tensor<3xf32>
    %3 = stablehlo.uniform_quantize %2 : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 0.0039043351715686275:-128>>
    %4 = stablehlo.dot_general %1, %3, contracting_dims = [0] x [0] : (tensor<3x!quant.uniform<i8:f32, 3.906250e-03:-128>>, tensor<3x!quant.uniform<i8:f32, 0.0039043351715686275:-128>>) -> tensor<!quant.uniform<i32:f32, 1.5251309263939951E-5>>
    %5 = stablehlo.uniform_quantize %4 : (tensor<!quant.uniform<i32:f32, 1.5251309263939951E-5>>) -> tensor<!quant.uniform<i8:f32, 0.0079394873450784123:-128>>
    %6 = stablehlo.uniform_dequantize %5 : (tensor<!quant.uniform<i8:f32, 0.0079394873450784123:-128>>) -> tensor<f32>
    %7 = stablehlo.custom_call @check.eq(%cst_1, %6) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %7 : tensor<i1>
  }
}
