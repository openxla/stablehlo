// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32> {mhlo.sharding = ""}, %arg2: tensor<?xi32> {mhlo.sharding = ""}) -> tensor<?x10x10xf32> {
    %0 = call @_take(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>
    return %0 : tensor<?x10x10xf32>
  }
  func.func private @_take(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>, %arg2: tensor<?xi32>) -> tensor<?x10x10xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = stablehlo.dynamic_broadcast_in_dim %arg2, %3, dims = [0] : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
    %5 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.constant dense<1> : tensor<1xi32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = stablehlo.dynamic_iota %8, dim = 0 : (tensor<2xi32>) -> tensor<?x1xi32>
    %10 = stablehlo.concatenate %9, %4, dim = 1 : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x2xi32>
    %11 = "stablehlo.gather"(%arg1, %10) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = dense<[1, 1, 10, 10]> : tensor<4xi64>} : (tensor<?x10x10x10xf32>, tensor<?x2xi32>) -> tensor<?x10x10xf32>
    return %11 : tensor<?x10x10xf32>
  }
}

