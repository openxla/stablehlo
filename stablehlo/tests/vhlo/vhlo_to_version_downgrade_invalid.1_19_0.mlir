// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.19.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.19.0}}
module {
  func.func @collective_broadcast_has_dynamic_root(%arg0: tensor<16x8xf32>, %arg1: tensor<1xi32>) -> tensor<16x8xf32> {
    // expected-error @+1 {{failed to legalize operation 'vhlo.collective_broadcast_v2' that was explicitly marked illegal}}
    %0 = "stablehlo.collective_broadcast"(%arg0, %arg1) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      has_dynamic_root
    } : (tensor<16x8xf32>, tensor<1xi32>) -> tensor<16x8xf32>
    func.return %0 : tensor<16x8xf32>
  }
}
