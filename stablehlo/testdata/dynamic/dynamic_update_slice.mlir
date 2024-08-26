// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.dynamic_update_slice %arg1, %arg1, %0, %1 : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i64>, tensor<i64>) -> tensor<?x4xf32>
    return %2 : tensor<?x4xf32>
  }
}

