// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x1xf32> {mhlo.sharding = ""}) -> tensor<?x1xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = stablehlo.dynamic_iota %3, dim = 0 : (tensor<2xi32>) -> tensor<?x1xui32>
    %5 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.constant dense<1> : tensor<1xi32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = stablehlo.dynamic_iota %8, dim = 1 : (tensor<2xi32>) -> tensor<?x1xui32>
    %10 = stablehlo.compare  EQ, %4, %9,  UNSIGNED : (tensor<?x1xui32>, tensor<?x1xui32>) -> tensor<?x1xi1>
    %11 = stablehlo.convert %10 : (tensor<?x1xi1>) -> tensor<?x1xf32>
    %12 = stablehlo.add %11, %arg1 : tensor<?x1xf32>
    return %12 : tensor<?x1xf32>
  }
}

