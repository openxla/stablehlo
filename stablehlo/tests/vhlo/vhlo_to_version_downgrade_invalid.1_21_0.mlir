// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.21.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.21.0}}
module {
  func.func @exp2(%arg0: tensor<f32>) -> tensor<f32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.exp2_v1' that was explicitly marked illegal}}
    %0 = "stablehlo.exp2"(%arg0) : (tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.21.0}}
module {
  func.func @log2(%arg0: tensor<f32>) -> tensor<f32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.log2_v1' that was explicitly marked illegal}}
    %0 = "stablehlo.log2"(%arg0) : (tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }
}
