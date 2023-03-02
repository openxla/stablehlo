// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3xbf16> {mhlo.sharding = ""}) -> tensor<?x3xbf16> {
    %0 = stablehlo.constant dense<0x7F80> : tensor<bf16>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x2x3xbf16>, tensor<bf16>) -> tensor<?x3xbf16>
     reducer(%arg2: tensor<bf16>, %arg3: tensor<bf16>)  {
      %2 = stablehlo.minimum %arg2, %arg3 : tensor<bf16>
      stablehlo.return %2 : tensor<bf16>
    }
    return %1 : tensor<?x3xbf16>
  }
}

