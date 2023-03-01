// RUN: echo "skipping CHLO test (see #1233 for details)"

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xf32> {mhlo.sharding = ""}, %arg2: tensor<?x20x20xf32> {mhlo.sharding = ""}) -> tensor<?x20x20xf32> {
    %0 = chlo.next_after %arg1, %arg2 : tensor<?x20x20xf32>, tensor<?x20x20xf32> -> tensor<?x20x20xf32>
    return %0 : tensor<?x20x20xf32>
  }
}

