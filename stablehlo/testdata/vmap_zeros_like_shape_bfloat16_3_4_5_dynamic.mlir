// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3x4x5xbf16> {mhlo.sharding = ""}) -> tensor<?x3x4x5xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<3> : tensor<1xi32>
    %4 = stablehlo.constant dense<4> : tensor<1xi32>
    %5 = stablehlo.constant dense<5> : tensor<1xi32>
    %6 = stablehlo.concatenate %2, %3, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %7 = stablehlo.dynamic_broadcast_in_dim %0, %6, dims = [] : (tensor<bf16>, tensor<4xi32>) -> tensor<?x3x4x5xbf16>
    return %7 : tensor<?x3x4x5xbf16>
  }
}

