// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2x3xf32> {mhlo.sharding = ""}) -> tensor<?x2x3xi1> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    %3 = stablehlo.constant dense<1> : tensor<1xi32>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %arg1, %4, dims = [0] : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x1x1xf32>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.constant dense<2> : tensor<1xi32>
    %9 = stablehlo.constant dense<3> : tensor<1xi32>
    %10 = stablehlo.concatenate %7, %8, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %5, %10, dims = [0, 1, 2] : (tensor<?x1x1xf32>, tensor<3xi32>) -> tensor<?x2x3xf32>
    %12 = stablehlo.compare  EQ, %11, %arg2,  FLOAT : (tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xi1>
    return %12 : tensor<?x2x3xi1>
  }
}

