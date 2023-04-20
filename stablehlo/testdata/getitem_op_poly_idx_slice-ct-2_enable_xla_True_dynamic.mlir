// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %2 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.constant dense<2> : tensor<1xi32>
    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %6 = "stablehlo.dynamic_gather"(%arg1, %1, %5) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], start_index_map = [1]>, indices_are_sorted = true} : (tensor<?x4xf32>, tensor<1xi64>, tensor<2xi32>) -> tensor<?x2xf32>
    return %6 : tensor<?x2xf32>
  }
}

