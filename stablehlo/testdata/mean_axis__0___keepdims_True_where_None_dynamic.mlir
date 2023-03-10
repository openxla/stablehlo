// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<1x8x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %6 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x8x4xf32>
    %5 = stablehlo.divide %2, %4 : tensor<1x8x4xf32>
    return %5 : tensor<1x8x4xf32>
  }
}

