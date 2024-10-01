// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.4.0' --verify-diagnostics --split-input-file %s

// expected-error @-3 {{failed to convert VHLO to v1.4.0}}
func.func @all_reduce_variadic(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  // expected-error @+1 {{failed to legalize operation 'vhlo.all_reduce_v2' that was explicitly marked illegal}}
  %0:2 = "stablehlo.all_reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.4.0}}
func.func @all_gather_variadic(%arg0: tensor<16x8xf32>, %arg1: tensor<16x8xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  // expected-error @+1 {{failed to legalize operation 'vhlo.all_gather_v2' that was explicitly marked illegal}}
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<16x8xf32>, tensor<16x8xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>)
  func.return %0#0, %0#1 : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

// expected-error @-3 {{failed to convert VHLO to v1.4.0}}
func.func @all_to_all_variadic(%arg0: tensor<4x16xf32>, %arg1: tensor<5x16xf32>) -> (tensor<16x4xf32>, tensor<20x4xf32>) {
  // expected-error @+1 {{failed to legalize operation 'vhlo.all_to_all_v2' that was explicitly marked illegal}}
  %0:2 = "stablehlo.all_to_all"(%arg0, %arg1) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>, tensor<5x16xf32>) -> (tensor<16x4xf32>, tensor<20x4xf32>)
  func.return %0#0, %0#1 : tensor<16x4xf32>, tensor<20x4xf32>
}
