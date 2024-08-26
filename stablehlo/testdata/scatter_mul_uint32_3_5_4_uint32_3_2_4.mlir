// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xui32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xui32>, tensor<3x2x4xui32>)
    %1 = call @expected() : () -> tensor<3x5x4xui32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<ui32>
      stablehlo.return %3 : tensor<ui32>
    }) : (tensor<3x5x4xui32>, tensor<2x1xi64>, tensor<3x2x4xui32>) -> tensor<3x5x4xui32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x5x4xui32>, tensor<3x5x4xui32>) -> ()
    return %2 : tensor<3x5x4xui32>
  }
  func.func private @inputs() -> (tensor<3x5x4xui32> {mhlo.layout_mode = "default"}, tensor<3x2x4xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1, 1, 0], [3, 0, 3, 2], [3, 1, 0, 1], [3, 1, 0, 0], [4, 4, 2, 1]], [[0, 0, 0, 1], [3, 5, 3, 0], [1, 1, 0, 0], [0, 0, 2, 2], [6, 0, 3, 1]], [[2, 5, 2, 3], [0, 2, 3, 1], [0, 4, 2, 2], [0, 5, 3, 3], [4, 4, 1, 1]]]> : tensor<3x5x4xui32>
    %c_0 = stablehlo.constant dense<[[[0, 2, 5, 0], [3, 1, 1, 0]], [[2, 3, 5, 4], [4, 4, 0, 3]], [[1, 4, 0, 2], [1, 1, 0, 0]]]> : tensor<3x2x4xui32>
    return %c, %c_0 : tensor<3x5x4xui32>, tensor<3x2x4xui32>
  }
  func.func private @expected() -> (tensor<3x5x4xui32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1, 1, 0], [0, 0, 15, 0], [3, 1, 0, 1], [3, 1, 0, 0], [4, 4, 2, 1]], [[0, 0, 0, 1], [24, 60, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [6, 0, 3, 1]], [[2, 5, 2, 3], [0, 8, 0, 0], [0, 4, 2, 2], [0, 5, 3, 3], [4, 4, 1, 1]]]> : tensor<3x5x4xui32>
    return %c : tensor<3x5x4xui32>
  }
}
