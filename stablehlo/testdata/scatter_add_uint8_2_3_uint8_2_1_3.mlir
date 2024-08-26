// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<1x3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<2x3xui8>, tensor<2x1x3xui8>)
    %1 = call @expected() : () -> tensor<2x3xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<2x3xui8>, tensor<1x3x1xi64>, tensor<2x1x3xui8>) -> tensor<2x3xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<2x3xui8>, tensor<2x3xui8>) -> ()
    return %2 : tensor<2x3xui8>
  }
  func.func private @inputs() -> (tensor<2x3xui8> {mhlo.layout_mode = "default"}, tensor<2x1x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 4, 1], [1, 3, 1]]> : tensor<2x3xui8>
    %c_0 = stablehlo.constant dense<[[[2, 2, 1]], [[3, 4, 0]]]> : tensor<2x1x3xui8>
    return %c, %c_0 : tensor<2x3xui8>, tensor<2x1x3xui8>
  }
  func.func private @expected() -> (tensor<2x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[3, 4, 6], [1, 3, 8]]> : tensor<2x3xui8>
    return %c : tensor<2x3xui8>
  }
}
