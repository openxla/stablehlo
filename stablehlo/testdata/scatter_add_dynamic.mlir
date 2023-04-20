// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<2x1xi32> {mhlo.sharding = ""}, %arg3: tensor<?x2xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = "stablehlo.scatter"(%arg1, %arg2, %arg3) ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %1 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = true} : (tensor<?x4xf32>, tensor<2x1xi32>, tensor<?x2xf32>) -> tensor<?x4xf32>
    return %0 : tensor<?x4xf32>
  }
}

