// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<2x1xi32> {mhlo.sharding = ""}, %arg3: tensor<?x2xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<3> : tensor<i64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %2 = stablehlo.constant dense<2147483647> : tensor<ui64>
    %3 = stablehlo.convert %2 : (tensor<ui64>) -> tensor<i64>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %5 = stablehlo.minimum %1, %4 : tensor<1xi64>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<1xi64>) -> tensor<2x1xi64>
    %7 = stablehlo.convert %arg2 : (tensor<2x1xi32>) -> tensor<2x1xi64>
    %8 = stablehlo.constant dense<0> : tensor<i64>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i64>) -> tensor<2x1xi64>
    %10 = stablehlo.clamp %9, %7, %6 : tensor<2x1xi64>
    %11 = "stablehlo.scatter"(%arg1, %10, %arg3) ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %12 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %12 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = true} : (tensor<?x4xf32>, tensor<2x1xi64>, tensor<?x2xf32>) -> tensor<?x4xf32>
    return %11 : tensor<?x4xf32>
  }
}

