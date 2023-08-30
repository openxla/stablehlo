// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3xf32> {mhlo.sharding = ""}) -> tensor<?x3x2xf32> {
    %0 = stablehlo.transpose %arg1, dims = [0, 2, 1] : (tensor<?x2x3xf32>) -> tensor<?x3x2xf32>
    return %0 : tensor<?x3x2xf32>
  }
}

