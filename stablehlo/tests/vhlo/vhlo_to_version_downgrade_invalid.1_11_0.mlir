// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.11.0' --verify-diagnostics --split-input-file %s

// expected-error @+1 {{failed to convert VHLO to v1.11.0}}
module {
func.func public @send_op(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  // expected-error @+1 {{failed to legalize operation 'vhlo.send_v2' that was explicitly marked illegal}}
  %0 = "stablehlo.send"(%arg0, %arg1) {
    source_target_pairs = dense<[[0,1],[1,2]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 2>,
    is_host_transfer = true
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
}

// -----

// expected-error @+1 {{failed to convert VHLO to v1.11.0}}
module {
func.func public @recv_op(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  // expected-error @+1 {{failed to legalize operation 'vhlo.recv_v2' that was explicitly marked illegal}}
  %0:2 = "stablehlo.recv"(%arg0) {
    source_target_pairs = dense<[[0,1],[1,2]]> : tensor<2x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 3>,
    is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}
}
