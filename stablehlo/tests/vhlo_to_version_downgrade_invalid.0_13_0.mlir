// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.13.0' --verify-diagnostics --split-input-file %s

func.func @uniform_dequantize(%arg: tensor<1x!quant.uniform<ui8:f16, 34.0:16>>) -> tensor<1xf16> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.uniform_dequantize_v1' that was explicitly marked illegal}}
  %0 = stablehlo.uniform_dequantize %arg : (tensor<1x!quant.uniform<ui8:f16, 34.0:16>>) -> tensor<1xf16>
  func.return %0 : tensor<1xf16>
}

// -----

func.func @uniform_quantize(%arg: tensor<1xf16>) -> tensor<1x!quant.uniform<ui8:f16, 34.0:16>> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.uniform_quantize_v1' that was explicitly marked illegal}}
  %0 = stablehlo.uniform_quantize %arg : (tensor<1xf16>) -> tensor<1x!quant.uniform<ui8:f16, 34.0:16>>
  func.return %0 : tensor<1x!quant.uniform<ui8:f16, 34.0:16>>
}
