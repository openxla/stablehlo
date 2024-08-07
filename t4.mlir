func.func @map(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = stablehlo.uniform_quantize %arg0 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
  %1 = stablehlo.uniform_dequantize %0 : (tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4x5xf32>
  %2 = stablehlo.uniform_quantize %arg1 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
  %3 = stablehlo.uniform_dequantize %2 : (tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4x5xf32>
  %4 = "stablehlo.map"(%1, %3) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %ret = stablehlo.constant dense<2.0> : tensor<f32>
    "stablehlo.return"(%ret) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = stablehlo.uniform_quantize %4 : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %5 : tensor<4x5x!quant.uniform<ui8:f32, 34.0:16>>
}
