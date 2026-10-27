// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.23.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.23.0}}
module {
  func.func @fma(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.fma_v1' that was explicitly marked illegal}}
    %0 = "stablehlo.fma"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }
}
