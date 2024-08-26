// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<?x?x5xf32> {mhlo.sharding = ""}) -> tensor<?x?x5xf32> {
    %0 = stablehlo.concatenate %arg2, %arg2, dim = 0 : (tensor<?x?x5xf32>, tensor<?x?x5xf32>) -> tensor<?x?x5xf32>
    return %0 : tensor<?x?x5xf32>
  }
}

