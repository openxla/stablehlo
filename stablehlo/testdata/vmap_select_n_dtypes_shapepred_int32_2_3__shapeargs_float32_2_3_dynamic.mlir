// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x3xi32> {mhlo.sharding = ""}, %arg2: tensor<?x2x3xf32> {mhlo.sharding = ""}, %arg3: tensor<?x2x3xf32> {mhlo.sharding = ""}, %arg4: tensor<?x2x3xf32> {mhlo.sharding = ""}) -> tensor<?x2x3xf32> {
    %0 = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.constant dense<3> : tensor<1xi32>
    %5 = stablehlo.concatenate %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %0, %5, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x2x3xi32>
    %7 = stablehlo.compare  LT, %arg1, %6,  SIGNED : (tensor<?x2x3xi32>, tensor<?x2x3xi32>) -> tensor<?x2x3xi1>
    %8 = stablehlo.constant dense<2> : tensor<i32>
    %9 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.constant dense<2> : tensor<1xi32>
    %12 = stablehlo.constant dense<3> : tensor<1xi32>
    %13 = stablehlo.concatenate %10, %11, %12, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = stablehlo.dynamic_broadcast_in_dim %8, %13, dims = [] : (tensor<i32>, tensor<3xi32>) -> tensor<?x2x3xi32>
    %15 = stablehlo.compare  LT, %arg1, %14,  SIGNED : (tensor<?x2x3xi32>, tensor<?x2x3xi32>) -> tensor<?x2x3xi1>
    %16 = stablehlo.select %15, %arg3, %arg4 : tensor<?x2x3xi1>, tensor<?x2x3xf32>
    %17 = stablehlo.select %7, %arg2, %16 : tensor<?x2x3xi1>, tensor<?x2x3xf32>
    return %17 : tensor<?x2x3xf32>
  }
}

