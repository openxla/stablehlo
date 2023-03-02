// RUN: echo "skipping CHLO test with dynamism ops from shape dialect (see #8 for details)"

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = chlo.lgamma %arg1 : tensor<?x20x20xbf16> -> tensor<?x20x20xbf16>
    return %0 : tensor<?x20x20xbf16>
  }
}

