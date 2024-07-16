// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x100x100xi8> {mhlo.sharding = ""}) -> tensor<?x100x100xui32> {
    %0 = stablehlo.convert %arg1 : (tensor<?x100x100xi8>) -> tensor<?x100x100xui32>
    return %0 : tensor<?x100x100xui32>
  }
}

