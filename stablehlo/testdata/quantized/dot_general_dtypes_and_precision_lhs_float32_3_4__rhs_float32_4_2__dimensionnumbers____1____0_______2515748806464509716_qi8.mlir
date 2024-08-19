module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[-0.988849937, -1.00252235, 1.64629126, 2.86447835], [4.66840506, -1.25282645, 1.20502043, -1.91219449], [0.263320625, 1.15289724, 0.175304011, -0.978201329]]> : tensor<3x4xf32>
    %cst_0 = stablehlo.constant dense<[[-0.117480382, -3.2312851], [0.462347686, -3.56523347], [4.92760229, 1.94133246], [0.446654767, 0.681284487]]> : tensor<4x2xf32>
    %cst_1 = stablehlo.constant dense<[[1.44528437, 1.68124914], [0.993018507, 0.993018507], [0.639071285, 0.176973596]]> : tensor<3x2xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<4x2xf32>) -> tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0] : (tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>, tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>) -> tensor<3x2xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
