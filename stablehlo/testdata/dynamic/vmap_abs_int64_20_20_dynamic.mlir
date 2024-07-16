// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xi64> {mhlo.sharding = ""}) -> tensor<?x20x20xi64> {
    %0 = stablehlo.abs %arg1 : tensor<?x20x20xi64>
    return %0 : tensor<?x20x20xi64>
  }
}

