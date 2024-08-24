// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[-1.02716792, -0.84180355, -1.21495497, 1.22473526], [1.533764, 3.72483683, -3.63489795, 1.59403718], [-2.40816736, -1.31785822, -1.78571689, 0.202816486]]]> : tensor<1x3x4xf32>
    %cst_0 = stablehlo.constant dense<[[[2.14761162, 2.50920868, -4.64897203], [1.72184336, -1.84444284, -1.14315307], [-1.28549385, 2.20468378, 1.76219583], [-3.49526167, -2.27656484, 2.42410755]]]> : tensor<1x4x3xf32>
    %cst_1 = stablehlo.constant dense<-0.558793128> : tensor<1xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [0], contracting_dims = [2, 1] x [1, 2] : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
