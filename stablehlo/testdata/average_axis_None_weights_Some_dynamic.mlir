// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg2 init: %0) across dimensions = [0, 1, 2] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %6 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }
    %2 = stablehlo.multiply %arg1, %arg2 : tensor<?x8x4xf32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.reduce(%2 init: %3) across dimensions = [0, 1, 2] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %6 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }
    %5 = stablehlo.divide %4, %1 : tensor<f32>
    return %5 : tensor<f32>
  }
}

