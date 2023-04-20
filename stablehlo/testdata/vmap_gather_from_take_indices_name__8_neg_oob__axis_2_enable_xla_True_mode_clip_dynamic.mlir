// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32> {mhlo.sharding = ""}, %arg2: tensor<?x2x2x1xi32> {mhlo.sharding = ""}) -> tensor<?x10x10x2x2x1xf32> {
    %0 = call @_take(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2x2x1xi32>) -> tensor<?x10x10x2x2x1xf32>
    return %0 : tensor<?x10x10x2x2x1xf32>
  }
  func.func private @_take(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>, %arg2: tensor<?x2x2x1xi32>) -> tensor<?x10x10x2x2x1xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<2> : tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.constant dense<1> : tensor<1xi32>
    %5 = stablehlo.constant dense<1> : tensor<1xi32>
    %6 = stablehlo.concatenate %1, %2, %3, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
    %7 = stablehlo.dynamic_broadcast_in_dim %arg2, %6, dims = [0, 1, 2, 3] : (tensor<?x2x2x1xi32>, tensor<5xi32>) -> tensor<?x2x2x1x1xi32>
    %8 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.constant dense<2> : tensor<1xi32>
    %11 = stablehlo.constant dense<2> : tensor<1xi32>
    %12 = stablehlo.constant dense<1> : tensor<1xi32>
    %13 = stablehlo.constant dense<1> : tensor<1xi32>
    %14 = stablehlo.concatenate %9, %10, %11, %12, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
    %15 = stablehlo.dynamic_iota %14, dim = 0 : (tensor<5xi32>) -> tensor<?x2x2x1x1xi32>
    %16 = stablehlo.concatenate %15, %7, dim = 4 : (tensor<?x2x2x1x1xi32>, tensor<?x2x2x1x1xi32>) -> tensor<?x2x2x1x2xi32>
    %17 = "stablehlo.gather"(%arg1, %16) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0, 3], start_index_map = [0, 3], index_vector_dim = 4>, slice_sizes = dense<[1, 10, 10, 1]> : tensor<4xi64>} : (tensor<?x10x10x10xf32>, tensor<?x2x2x1x2xi32>) -> tensor<?x10x10x2x2x1xf32>
    return %17 : tensor<?x10x10x2x2x1xf32>
  }
}

