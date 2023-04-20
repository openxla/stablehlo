// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.dynamic_update_slice %arg1, %arg1, %0, %1 : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i64>, tensor<i64>) -> tensor<?x4xf32>
    return %2 : tensor<?x4xf32>
  }
}

