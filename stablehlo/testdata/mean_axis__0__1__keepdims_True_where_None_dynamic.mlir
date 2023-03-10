// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<1x1x4xf32> {
    %0 = stablehlo.constant dense<8> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%arg1 init: %2) across dimensions = [0, 1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %8 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %8 : tensor<f32>
    }
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4xf32>
    %5 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1x4xf32>
    %7 = stablehlo.divide %4, %6 : tensor<1x1x4xf32>
    return %7 : tensor<1x1x4xf32>
  }
}

