module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[0.30003655, 3.43623924, -5.00657272, -5.54167175], [-0.0709157884, 5.499670e+00, 3.123830e-01, 0.863251984], [2.37204719, 0.134123445, -0.936426699, 1.57510769]]> : tensor<3x4xf32>
    %cst_0 = stablehlo.constant dense<[[1.22435701, 1.81742287], [-2.47757602, 0.965225696], [-3.48883629, 0.23426415], [-2.27880955, 0.550582886]]> : tensor<4x2xf32>
    %cst_1 = stablehlo.constant dense<[[0.304787844, 1.26831079], [0.000000e+00, 1.51410735], [0.993018507, 1.68124914]]> : tensor<3x2xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<4x2xf32>) -> tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>
    %2 = stablehlo.dot_general %1, %0, contracting_dims = [1] x [0], precision = [HIGH, HIGH] : (tensor<3x4x!quant.uniform<i8:f32, 0.0039170005742241356:-128>>, tensor<4x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<3x2x!quant.uniform<i32:f32, 1.5343835624351492E-5>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<3x2x!quant.uniform<i8:f32, 0.0098318660960477945:-128>>) -> tensor<3x2xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
