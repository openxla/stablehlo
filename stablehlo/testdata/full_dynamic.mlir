// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x1xf32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %0, %4, dims = [] : (tensor<f64>, tensor<2xi32>) -> tensor<?x2xf64>
    %6 = stablehlo.convert %5 : (tensor<?x2xf64>) -> tensor<?x2xf32>
    %7 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.constant dense<2> : tensor<1xi32>
    %10 = stablehlo.concatenate %8, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %arg1, %10, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x2xf32>
    %12 = stablehlo.add %6, %11 : tensor<?x2xf32>
    return %12 : tensor<?x2xf32>
  }
}

