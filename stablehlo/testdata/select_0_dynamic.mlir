// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3xf32> {mhlo.sharding = ""}) -> tensor<?x3xf32> {
    %0 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<3> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %0, %4, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x3xf32>
    %6 = stablehlo.compare  GT, %arg1, %5,  FLOAT : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
    %7 = stablehlo.select %6, %arg1, %arg1 : tensor<?x3xi1>, tensor<?x3xf32>
    return %7 : tensor<?x3xf32>
  }
}

