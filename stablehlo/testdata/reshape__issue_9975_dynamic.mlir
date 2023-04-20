// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?xf32> {
    %0 = stablehlo.constant dense<4> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.dynamic_reshape %arg1, %3 : (tensor<?x4xf32>, tensor<1xi32>) -> tensor<?xf32>
    return %4 : tensor<?xf32>
  }
}

