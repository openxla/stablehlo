// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x30xbf16> {mhlo.sharding = ""}) -> tensor<?x20x30xbf16> {
    %0 = stablehlo.multiply %arg1, %arg1 : tensor<?x20x30xbf16>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.constant dense<20> : tensor<1xi32>
    %5 = stablehlo.constant dense<30> : tensor<1xi32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = stablehlo.dynamic_broadcast_in_dim %1, %6, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x30xbf16>
    %8 = stablehlo.divide %7, %0 : tensor<?x20x30xbf16>
    return %8 : tensor<?x20x30xbf16>
  }
}

