// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=0.15.0' --verify-diagnostics --split-input-file %s

func.func @default_collective_broadcast(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.collective_broadcast_v1' that was explicitly marked illegal}}
  %0 = "stablehlo.collective_broadcast"(%arg0) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}
