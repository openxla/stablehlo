// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xui64> {mhlo.sharding = ""}) -> tensor<?x20x20xui64> {
    %0 = stablehlo.constant dense<0> : tensor<ui64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<20> : tensor<1xi32>
    %4 = stablehlo.constant dense<20> : tensor<1xi32>
    %5 = stablehlo.concatenate %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %0, %5, dims = [] : (tensor<ui64>, tensor<3xi32>) -> tensor<?x20x20xui64>
    %7 = stablehlo.compare  EQ, %arg1, %6,  UNSIGNED : (tensor<?x20x20xui64>, tensor<?x20x20xui64>) -> tensor<?x20x20xi1>
    %8 = stablehlo.constant dense<0> : tensor<ui64>
    %9 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.constant dense<20> : tensor<1xi32>
    %12 = stablehlo.constant dense<20> : tensor<1xi32>
    %13 = stablehlo.concatenate %10, %11, %12, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = stablehlo.dynamic_broadcast_in_dim %8, %13, dims = [] : (tensor<ui64>, tensor<3xi32>) -> tensor<?x20x20xui64>
    %15 = stablehlo.constant dense<1> : tensor<ui64>
    %16 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.constant dense<20> : tensor<1xi32>
    %19 = stablehlo.constant dense<20> : tensor<1xi32>
    %20 = stablehlo.concatenate %17, %18, %19, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %21 = stablehlo.dynamic_broadcast_in_dim %15, %20, dims = [] : (tensor<ui64>, tensor<3xi32>) -> tensor<?x20x20xui64>
    %22 = stablehlo.select %7, %14, %21 : tensor<?x20x20xi1>, tensor<?x20x20xui64>
    return %22 : tensor<?x20x20xui64>
  }
}

