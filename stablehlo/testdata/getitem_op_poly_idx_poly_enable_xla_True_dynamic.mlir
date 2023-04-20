// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?xi32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.dynamic_broadcast_in_dim %1, %3, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %5 = stablehlo.compare  LT, %arg2, %4,  SIGNED : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.dynamic_broadcast_in_dim %0, %7, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %9 = stablehlo.add %arg2, %8 : tensor<?xi32>
    %10 = stablehlo.select %5, %9, %arg2 : tensor<?xi1>, tensor<?xi32>
    %11 = stablehlo.convert %10 : (tensor<?xi32>) -> tensor<?xi64>
    %12 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %13 = stablehlo.reshape %12 : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.constant dense<1> : tensor<1xi32>
    %15 = stablehlo.concatenate %13, %14, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = stablehlo.dynamic_broadcast_in_dim %11, %15, dims = [0] : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %17 = "stablehlo.gather"(%arg1, %16) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<?x1xi64>) -> tensor<?x4xf32>
    return %17 : tensor<?x4xf32>
  }
}

