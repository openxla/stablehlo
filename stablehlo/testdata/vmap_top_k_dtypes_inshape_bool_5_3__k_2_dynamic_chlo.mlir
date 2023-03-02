// RUN: echo "skipping CHLO dynamic TopK test (see #1255 for details)"

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5x3xi1> {mhlo.sharding = ""}) -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>) {
    %values, %indices = chlo.top_k(%arg1, k = 2) : tensor<?x5x3xi1> -> (tensor<?x5x2xi1>, tensor<?x5x2xi32>)
    return %values, %indices : tensor<?x5x2xi1>, tensor<?x5x2xi32>
  }
}

