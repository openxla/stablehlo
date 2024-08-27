// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<1x3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<2x3xi16>, tensor<2x1x3xi16>)
    %1 = call @expected() : () -> tensor<2x3xi16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<2x3xi16>, tensor<1x3x1xi64>, tensor<2x1x3xi16>) -> tensor<2x3xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<2x3xi16>, tensor<2x3xi16>) -> ()
    return %2 : tensor<2x3xi16>
  }
  func.func private @inputs() -> (tensor<2x3xi16> {mhlo.layout_mode = "default"}, tensor<2x1x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, 2, -2], [2, -1, -3]]> : tensor<2x3xi16>
    %c_0 = stablehlo.constant dense<[[[-3, 0, 1]], [[-1, -2, -1]]]> : tensor<2x1x3xi16>
    return %c, %c_0 : tensor<2x3xi16>, tensor<2x1x3xi16>
  }
  func.func private @expected() -> (tensor<2x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, 2, -3], [2, -1, -3]]> : tensor<2x3xi16>
    return %c : tensor<2x3xi16>
  }
}
