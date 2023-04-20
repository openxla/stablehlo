// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2xf32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = stablehlo.remainder %arg1, %arg2 : tensor<?x2xf32>
    return %0 : tensor<?x2xf32>
  }
}

