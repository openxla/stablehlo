// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?xf32> {mhlo.sharding = ""}, %arg2: tensor<?xf32> {mhlo.sharding = ""}) -> tensor<?xi1> {
    %0 = stablehlo.compare  GE, %arg1, %arg2,  FLOAT : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    return %0 : tensor<?xi1>
  }
}

