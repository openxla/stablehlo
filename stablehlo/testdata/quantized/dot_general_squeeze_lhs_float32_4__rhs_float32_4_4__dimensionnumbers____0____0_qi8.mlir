module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[-2.67958117, 4.91505384, -2.93944049, -0.189866632]> : tensor<4xf32>
    %cst_0 = stablehlo.constant dense<[[-0.876051664, 2.81679201, 1.48077691, 1.10807765], [-1.83372617, 1.35355616, 3.68328929, -4.30171204], [-6.15009593, -5.9722824, -0.454436153, -1.66895545], [1.09934378, 5.87006092, -3.10807371, 0.333222806]]> : tensor<4x4xf32>
    %cst_1 = stablehlo.constant dense<[0.000000e+00, 0.994613945, 0.994613945, 0.000000e+00]> : tensor<4xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<4x4xf32>) -> tensor<4x4x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [0] x [0] : (tensor<4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>, tensor<4x4x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<4x!quant.uniform<i32:f32, 1.5343835624351492E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<4x!quant.uniform<i32:f32, 1.5343835624351492E-5>>) -> tensor<4x!quant.uniform<i8:f32, 0.0082884498670989393:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<4x!quant.uniform<i8:f32, 0.0082884498670989393:-128>>) -> tensor<4xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
