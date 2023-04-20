// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?xi32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_broadcast_in_dim %0, %2, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %4 = stablehlo.compare  LT, %arg2, %3,  SIGNED : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %5 = stablehlo.constant dense<3> : tensor<i32>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.dynamic_broadcast_in_dim %5, %7, dims = [] : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %9 = stablehlo.add %arg2, %8 : tensor<?xi32>
    %10 = stablehlo.select %4, %9, %arg2 : tensor<?xi1>, tensor<?xi32>
    %11 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.constant dense<1> : tensor<1xi32>
    %14 = stablehlo.concatenate %12, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.dynamic_broadcast_in_dim %10, %14, dims = [0] : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
    %16 = "stablehlo.gather"(%arg1, %15) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xf32>, tensor<?x1xi32>) -> tensor<?x4xf32>
    return %16 : tensor<?x4xf32>
  }
}

