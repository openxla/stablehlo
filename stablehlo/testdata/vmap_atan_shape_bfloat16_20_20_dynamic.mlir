// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.concatenate %3, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %6 = stablehlo.atan2 %arg1, %5 : tensor<?x20x20xbf16>
    return %6 : tensor<?x20x20xbf16>
  }
}

