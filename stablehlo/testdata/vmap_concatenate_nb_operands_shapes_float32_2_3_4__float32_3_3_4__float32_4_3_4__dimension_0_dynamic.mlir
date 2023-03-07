// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x3x3x4xf32> {mhlo.sharding = ""}, %arg3: tensor<?x4x3x4xf32> {mhlo.sharding = ""}) -> tensor<?x9x3x4xf32> {
    %0 = stablehlo.concatenate %arg1, %arg2, %arg3, dim = 1 : (tensor<?x2x3x4xf32>, tensor<?x3x3x4xf32>, tensor<?x4x3x4xf32>) -> tensor<?x9x3x4xf32>
    return %0 : tensor<?x9x3x4xf32>
  }
}

