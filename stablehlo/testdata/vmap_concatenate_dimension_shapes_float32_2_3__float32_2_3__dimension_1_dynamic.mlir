// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2x3xf32> {mhlo.sharding = ""}) -> tensor<?x2x6xf32> {
    %0 = stablehlo.concatenate %arg1, %arg2, dim = 2 : (tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x6xf32>
    return %0 : tensor<?x2x6xf32>
  }
}

