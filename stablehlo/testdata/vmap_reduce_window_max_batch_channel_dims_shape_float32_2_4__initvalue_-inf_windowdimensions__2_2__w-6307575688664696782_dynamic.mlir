// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x4xf32> {mhlo.sharding = ""}) -> tensor<?x1x3xf32> {
    %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.reduce_window"(%arg1, %1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %3 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) {window_dimensions = dense<[1, 2, 2]> : tensor<3xi64>} : (tensor<?x2x4xf32>, tensor<f32>) -> tensor<?x1x3xf32>
    return %2 : tensor<?x1x3xf32>
  }
}

