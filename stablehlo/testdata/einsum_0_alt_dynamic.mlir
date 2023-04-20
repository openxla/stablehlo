// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?xf32> {
    %0 = call @_einsum(%arg0, %arg1) : (tensor<i64>, tensor<?x4xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @_einsum(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>) -> tensor<?xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x4xf32>, tensor<f32>) -> tensor<?xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }
    return %1 : tensor<?xf32>
  }
}

