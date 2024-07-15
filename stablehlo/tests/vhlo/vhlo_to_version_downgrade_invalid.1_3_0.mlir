// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.3.0' --verify-diagnostics --split-input-file %s

func.func @op_tan(%arg0: tensor<f32>) -> tensor<f32> {
// expected-error @+1 {{failed to legalize operation 'vhlo.tan_v1' that was explicitly marked illegal}}
  %0 = "stablehlo.tan"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}