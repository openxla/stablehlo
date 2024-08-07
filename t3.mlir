module {
  func.func @add(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
    %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %2 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
    %4 = stablehlo.add %1, %3 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = stablehlo.uniform_quantize %4 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %7 = stablehlo.maximum %2, %5 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %8 = stablehlo.maximum %3, %4 : tensor<16x16xf32>
    %9 = stablehlo.uniform_quantize %8 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    %10 = stablehlo.add %7, %9 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    func.return %10 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
  }
}

// func.func @add(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>> {
//  %0 = stablehlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %1 = stablehlo.uniform_dequantize %0 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
//  %2 = stablehlo.uniform_quantize %arg1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %3 = stablehlo.uniform_dequantize %2 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
//  %4 = stablehlo.add %1, %3 : tensor<16x16xf32>
//  %5 = stablehlo.add %0, %2 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %6 = stablehlo.uniform_quantize %4 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %7 = stablehlo.maximum %2, %5 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %8 = stablehlo.maximum %3, %4 : tensor<16x16xf32>
//  %9 = stablehlo.uniform_quantize %8 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  %10 = stablehlo.add %7, %9 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//  return %10 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
//}
