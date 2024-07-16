// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x6xbf16> {mhlo.sharding = ""}) -> tensor<?x3x5xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = "stablehlo.reduce_window"(%arg1, %0) ({
    ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %2 = stablehlo.multiply %arg2, %arg3 : tensor<bf16>
      stablehlo.return %2 : tensor<bf16>
    }) {window_dimensions = array<i64: 1, 2, 2>} : (tensor<?x4x6xbf16>, tensor<bf16>) -> tensor<?x3x5xbf16>
    return %1 : tensor<?x3x5xbf16>
  }
}

