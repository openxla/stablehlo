// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?xbf16> {mhlo.sharding = ""}) -> tensor<?xbf16> {
    %0 = stablehlo.reduce_precision %arg1, format = e11m52 : tensor<?xbf16>
    return %0 : tensor<?xbf16>
  }
}

