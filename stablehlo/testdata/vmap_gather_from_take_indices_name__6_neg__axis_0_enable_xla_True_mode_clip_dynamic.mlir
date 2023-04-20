// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2x2xi32> {mhlo.sharding = ""}) -> tensor<?x2x2x10x10xf32> {
    %0 = call @_take(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2x2xi32>) -> tensor<?x2x2x10x10xf32>
    return %0 : tensor<?x2x2x10x10xf32>
  }
  func.func private @_take(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>, %arg2: tensor<?x2x2xi32>) -> tensor<?x2x2x10x10xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<2> : tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.constant dense<1> : tensor<1xi32>
    %5 = stablehlo.concatenate %1, %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %arg2, %5, dims = [0, 1, 2] : (tensor<?x2x2xi32>, tensor<4xi32>) -> tensor<?x2x2x1xi32>
    %7 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.constant dense<2> : tensor<1xi32>
    %10 = stablehlo.constant dense<2> : tensor<1xi32>
    %11 = stablehlo.constant dense<1> : tensor<1xi32>
    %12 = stablehlo.concatenate %8, %9, %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %13 = stablehlo.dynamic_iota %12, dim = 0 : (tensor<4xi32>) -> tensor<?x2x2x1xi32>
    %14 = stablehlo.concatenate %13, %6, dim = 3 : (tensor<?x2x2x1xi32>, tensor<?x2x2x1xi32>) -> tensor<?x2x2x2xi32>
    %15 = "stablehlo.gather"(%arg1, %14) {dimension_numbers = #stablehlo.gather<offset_dims = [3, 4], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 3>, slice_sizes = dense<[1, 1, 10, 10]> : tensor<4xi64>} : (tensor<?x10x10x10xf32>, tensor<?x2x2x2xi32>) -> tensor<?x2x2x10x10xf32>
    return %15 : tensor<?x2x2x10x10xf32>
  }
}

