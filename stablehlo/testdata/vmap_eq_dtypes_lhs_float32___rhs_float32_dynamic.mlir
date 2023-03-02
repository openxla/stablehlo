// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?xf32> {mhlo.sharding = ""}, %arg2: tensor<?xf32> {mhlo.sharding = ""}) -> tensor<?xi1> {
    %0 = stablehlo.compare  EQ, %arg1, %arg2,  FLOAT : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    return %0 : tensor<?xi1>
  }
}

