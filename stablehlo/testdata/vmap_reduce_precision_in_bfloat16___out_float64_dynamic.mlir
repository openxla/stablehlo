// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?xbf16> {mhlo.sharding = ""}) -> tensor<?xbf16> {
    %0 = stablehlo.reduce_precision %arg1, format = e11m52 : tensor<?xbf16>
    return %0 : tensor<?xbf16>
  }
}

