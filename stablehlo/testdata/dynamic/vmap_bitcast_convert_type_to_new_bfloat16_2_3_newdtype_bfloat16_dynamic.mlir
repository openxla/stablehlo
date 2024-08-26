// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3xbf16> {mhlo.sharding = ""}) -> tensor<?x2x3xbf16> {
    %0 = stablehlo.bitcast_convert %arg1 : (tensor<?x2x3xbf16>) -> tensor<?x2x3xbf16>
    return %0 : tensor<?x2x3xbf16>
  }
}

