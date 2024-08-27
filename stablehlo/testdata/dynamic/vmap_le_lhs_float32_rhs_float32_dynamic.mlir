// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?xf32> {mhlo.sharding = ""}, %arg2: tensor<?xf32> {mhlo.sharding = ""}) -> tensor<?xi1> {
    %0 = stablehlo.compare  LE, %arg1, %arg2,  FLOAT : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    return %0 : tensor<?xi1>
  }
}

