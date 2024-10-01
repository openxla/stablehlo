// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.17.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v0.17.0}}
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @type_per_axis_quantization(%arg0: tensor<2x!quant.uniform<i8:f32:0, {34.0:16, 34.0:16}>>) -> tensor<2x!quant.uniform<i8:f32:0, {34.0:16, 34.0:16}>> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<2x!quant.uniform<i8:f32:0, {34.0:16, 34.0:16}>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32:0, {34.0:16, 34.0:16}>>
}
