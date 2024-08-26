// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xi32>, tensor<3x2x4xi32>)
    %1 = call @expected() : () -> tensor<3x5x4xi32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) : (tensor<3x5x4xi32>, tensor<2x1xi64>, tensor<3x2x4xi32>) -> tensor<3x5x4xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x5x4xi32>, tensor<3x5x4xi32>) -> ()
    return %2 : tensor<3x5x4xi32>
  }
  func.func private @inputs() -> (tensor<3x5x4xi32> {mhlo.layout_mode = "default"}, tensor<3x2x4xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1, -4, -1], [1, 3, -2, 1], [1, -1, -1, -4], [0, 4, -1, -3], [-2, 0, 2, 3]], [[3, -3, 0, 1], [1, -1, -3, 1], [0, 1, 0, 2], [0, -1, 2, 4], [-1, 7, 0, 2]], [[4, -3, 0, -4], [1, 1, 0, 3], [-1, 6, 2, -1], [2, 0, -1, 5], [0, -5, 0, -2]]]> : tensor<3x5x4xi32>
    %c_0 = stablehlo.constant dense<[[[0, -2, 5, 0], [0, -2, 0, -1]], [[3, 0, 4, 0], [-1, 4, -4, 2]], [[-2, 3, -4, 1], [-2, -3, 0, -2]]]> : tensor<3x2x4xi32>
    return %c, %c_0 : tensor<3x5x4xi32>, tensor<3x2x4xi32>
  }
  func.func private @expected() -> (tensor<3x5x4xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1, -4, -1], [0, -2, -2, -1], [1, -1, -1, -4], [0, 4, -1, -3], [-2, 0, 2, 3]], [[3, -3, 0, 1], [-1, -1, -4, 0], [0, 1, 0, 2], [0, -1, 2, 4], [-1, 7, 0, 2]], [[4, -3, 0, -4], [-2, -3, -4, -2], [-1, 6, 2, -1], [2, 0, -1, 5], [0, -5, 0, -2]]]> : tensor<3x5x4xi32>
    return %c : tensor<3x5x4xi32>
  }
}
