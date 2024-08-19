module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[4.54189587, -0.56804496, 2.11517882]> : tensor<3xf32>
    %cst_0 = stablehlo.constant dense<[[4.05453825, -1.44036746, 4.79499197, 2.67171788, 0.616522252, -4.48888302], [4.79667568, -4.19831371, -1.44450212, 2.85810256, -0.367232859, 1.95318091], [-1.62081826, -5.7897296, 1.62217569, -0.311252445, 1.74422383, -0.186609924]]> : tensor<3x6xf32>
    %cst_1 = stablehlo.constant dense<[0.996517777, 0.000000e+00, 1.84139156, 0.996517777, 1.61031497, 0.000000e+00]> : tensor<6xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x6xf32>) -> tensor<3x6x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<3xf32>) -> tensor<3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [0] x [0] : (tensor<3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<3x6x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>) -> tensor<6x!quant.uniform<i32:f32, 1.5343835624351492E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<6x!quant.uniform<i32:f32, 1.5343835624351492E-5>>) -> tensor<6x!quant.uniform<i8:f32, 0.0072211433859432446:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<6x!quant.uniform<i8:f32, 0.0072211433859432446:-128>>) -> tensor<6xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<6xf32>, tensor<6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
