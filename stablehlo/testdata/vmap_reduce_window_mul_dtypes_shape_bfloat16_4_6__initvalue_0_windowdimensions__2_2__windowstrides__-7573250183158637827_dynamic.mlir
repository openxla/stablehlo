// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x6xbf16> {mhlo.sharding = ""}) -> tensor<?x3x5xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = "stablehlo.reduce_window"(%arg1, %0) ({
    ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %2 = stablehlo.multiply %arg2, %arg3 : tensor<bf16>
      stablehlo.return %2 : tensor<bf16>
    }) {window_dimensions = dense<[1, 2, 2]> : tensor<3xi64>} : (tensor<?x4x6xbf16>, tensor<bf16>) -> tensor<?x3x5xbf16>
    return %1 : tensor<?x3x5xbf16>
  }
}

