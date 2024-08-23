// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[3.29589868, 1.23571205, 1.77729869, -1.18627894], [-0.334112585, -0.701673209, 1.37458372, 1.33988047], [-1.23167276, -1.47567272, -5.089730e+00, 4.67841911]]]> : tensor<1x3x4xf32>
    %cst_0 = stablehlo.constant dense<[[[-1.98014688, 6.42148352, -4.60602903], [0.598645091, -1.56104231, 1.48972511], [2.90995979, -0.593875766, -4.25129318], [-1.8781029, 1.97643542, 2.91169143]]]> : tensor<1x4x3xf32>
    %cst_1 = stablehlo.constant dense<0.760579526> : tensor<1xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [0], contracting_dims = [2, 1] x [1, 2], precision = [HIGH, HIGH] : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}
