// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.18.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.18.0}}
module {
  func.func @collective_reduce(%arg0: tensor<f32>) -> tensor<f32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.collective_reduce_v1' that was explicitly marked illegal}}
    %0 = "stablehlo.collective_reduce"(%arg0) ({
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }
}
