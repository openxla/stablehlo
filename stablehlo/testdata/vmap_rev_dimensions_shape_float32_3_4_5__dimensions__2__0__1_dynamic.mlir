// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3x4x5xf32> {mhlo.sharding = ""}) -> tensor<?x3x4x5xf32> {
    %0 = stablehlo.reverse %arg1, dims = [3, 1, 2] : tensor<?x3x4x5xf32>
    return %0 : tensor<?x3x4x5xf32>
  }
}

