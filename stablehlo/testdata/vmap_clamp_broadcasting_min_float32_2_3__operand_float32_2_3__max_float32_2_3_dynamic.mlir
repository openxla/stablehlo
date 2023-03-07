// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2x3xf32> {mhlo.sharding = ""}, %arg3: tensor<?x2x3xf32> {mhlo.sharding = ""}) -> tensor<?x2x3xf32> {
    %0 = stablehlo.clamp %arg1, %arg2, %arg3 : tensor<?x2x3xf32>
    return %0 : tensor<?x2x3xf32>
  }
}

