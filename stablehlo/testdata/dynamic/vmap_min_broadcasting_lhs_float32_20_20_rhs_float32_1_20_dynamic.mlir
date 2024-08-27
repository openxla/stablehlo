// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xf32> {mhlo.sharding = ""}, %arg2: tensor<?x1x20xf32> {mhlo.sharding = ""}) -> tensor<?x20x20xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<20> : tensor<1xi32>
    %3 = stablehlo.constant dense<20> : tensor<1xi32>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %arg2, %4, dims = [0, 1, 2] : (tensor<?x1x20xf32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %6 = stablehlo.minimum %arg1, %5 : tensor<?x20x20xf32>
    return %6 : tensor<?x20x20xf32>
  }
}

