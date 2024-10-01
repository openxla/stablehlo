// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.18.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v0.18.0}}
func.func @composite(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.composite_v1' that was explicitly marked illegal}}
  %0 = "stablehlo.composite"(%arg0) {
    name = "stablehlo.composite_target",
    decomposition = @composite_target
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

func.func @composite_target(%arg0: tensor<f32>) -> tensor<f32> {
  func.return %arg0 : tensor<f32>
}
