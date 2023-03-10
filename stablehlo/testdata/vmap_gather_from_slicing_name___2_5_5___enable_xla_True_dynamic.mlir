// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32> {mhlo.sharding = ""}) -> tensor<?x3x10xf32> {
    %0 = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<5> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.constant dense<3> : tensor<1xi32>
    %8 = stablehlo.constant dense<1> : tensor<1xi32>
    %9 = stablehlo.constant dense<10> : tensor<1xi32>
    %10 = stablehlo.concatenate %6, %7, %8, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %11 = "stablehlo.dynamic_gather"(%arg1, %4, %10) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [2], start_index_map = [1, 2]>, indices_are_sorted = true} : (tensor<?x10x10x10xf32>, tensor<2xi32>, tensor<4xi32>) -> tensor<?x3x10xf32>
    return %11 : tensor<?x3x10xf32>
  }
}

