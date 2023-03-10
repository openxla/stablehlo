// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<?x4x?x6x7xf32> {mhlo.sharding = ""}) -> tensor<2x?xf32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
    %1 = stablehlo.constant dense<84> : tensor<i64>
    %2 = stablehlo.multiply %0, %1 : tensor<i64>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = stablehlo.dynamic_reshape %arg2, %6 : (tensor<?x4x?x6x7xf32>, tensor<2xi32>) -> tensor<2x?xf32>
    return %7 : tensor<2x?xf32>
  }
}

