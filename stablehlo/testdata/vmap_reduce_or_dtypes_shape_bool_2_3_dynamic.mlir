// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3xi1> {mhlo.sharding = ""}) -> tensor<?x3xi1> {
    %0 = stablehlo.constant dense<false> : tensor<i1>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x2x3xi1>, tensor<i1>) -> tensor<?x3xi1>
     reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
      %2 = stablehlo.or %arg2, %arg3 : tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }
    return %1 : tensor<?x3xi1>
  }
}

