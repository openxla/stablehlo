// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x1x2xf32> {mhlo.sharding = ""}, %arg2: tensor<?x1x2xi64> {mhlo.sharding = ""}) -> tensor<?x1xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    %3 = stablehlo.constant dense<1> : tensor<1xi32>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_iota %4, dim = 0 : (tensor<3xi32>) -> tensor<?x1x1xi64>
    %6 = stablehlo.concatenate %5, %arg2, dim = 2 : (tensor<?x1x1xi64>, tensor<?x1x2xi64>) -> tensor<?x1x3xi64>
    %7 = "stablehlo.gather"(%arg1, %6) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 2>, slice_sizes = dense<1> : tensor<3xi64>} : (tensor<?x1x2xf32>, tensor<?x1x3xi64>) -> tensor<?x1xf32>
    return %7 : tensor<?x1xf32>
  }
}

