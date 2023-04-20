// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<2x?x4xf32> {mhlo.sharding = ""}) -> tensor<2x?x4xf32> {
    %0 = stablehlo.constant dense<1> : tensor<1xi32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<4> : tensor<1xi32>
    %4 = stablehlo.concatenate %0, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %arg1, %4, dims = [1, 2] : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<1x?x4xf32>
    %6 = stablehlo.constant dense<2> : tensor<1xi32>
    %7 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.constant dense<4> : tensor<1xi32>
    %10 = stablehlo.concatenate %6, %8, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %5, %10, dims = [0, 1, 2] : (tensor<1x?x4xf32>, tensor<3xi32>) -> tensor<2x?x4xf32>
    %12 = stablehlo.add %11, %arg2 : tensor<2x?x4xf32>
    return %12 : tensor<2x?x4xf32>
  }
}

