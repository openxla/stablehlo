// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x2x3xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x2x3xi32>, tensor<1x3xi32>)
    %1 = call @expected() : () -> tensor<1x2x3xi32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) : (tensor<1x2x3xi32>, tensor<1xi64>, tensor<1x3xi32>) -> tensor<1x2x3xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x2x3xi32>, tensor<1x2x3xi32>) -> ()
    return %2 : tensor<1x2x3xi32>
  }
  func.func private @inputs() -> (tensor<1x2x3xi32> {mhlo.layout_mode = "default"}, tensor<1x3xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, -2, -3], [0, 0, -3]]]> : tensor<1x2x3xi32>
    %c_0 = stablehlo.constant dense<[[-2, -1, 0]]> : tensor<1x3xi32>
    return %c, %c_0 : tensor<1x2x3xi32>, tensor<1x3xi32>
  }
  func.func private @expected() -> (tensor<1x2x3xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, -2, -3], [0, 0, 0]]]> : tensor<1x2x3xi32>
    return %c : tensor<1x2x3xi32>
  }
}
