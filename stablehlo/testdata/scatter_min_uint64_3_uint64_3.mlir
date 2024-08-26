// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3xui64>, tensor<3xui64>)
    %1 = call @expected() : () -> tensor<3xui64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<ui64>, %arg1: tensor<ui64>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<ui64>
      stablehlo.return %3 : tensor<ui64>
    }) : (tensor<3xui64>, tensor<3x1xi64>, tensor<3xui64>) -> tensor<3xui64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3xui64>, tensor<3xui64>) -> ()
    return %2 : tensor<3xui64>
  }
  func.func private @inputs() -> (tensor<3xui64> {mhlo.layout_mode = "default"}, tensor<3xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[4, 3, 1]> : tensor<3xui64>
    %c_0 = stablehlo.constant dense<[4, 1, 1]> : tensor<3xui64>
    return %c, %c_0 : tensor<3xui64>, tensor<3xui64>
  }
  func.func private @expected() -> (tensor<3xui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<3xui64>
    return %c : tensor<3xui64>
  }
}
