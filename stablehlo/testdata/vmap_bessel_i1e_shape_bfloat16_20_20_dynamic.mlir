// RUN: echo "skipping CHLO test with dynamism ops from shape dialect (see #1233 for details)"

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = chlo.bessel_i1e %arg1 : tensor<?x20x20xbf16> -> tensor<?x20x20xbf16>
    return %0 : tensor<?x20x20xbf16>
  }
}

