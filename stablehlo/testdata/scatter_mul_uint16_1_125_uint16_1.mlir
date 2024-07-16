// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %1 = call @expected() : () -> tensor<1x125xui16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    return %2 : tensor<1x125xui16>
  }
  func.func private @inputs() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}, tensor<1xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x01000000050001000200020001000200070001000400000003000300000000000000000002000300070000000200020000000000010003000200020004000100000003000300000003000100020000000600050001000000020002000000070000000000020002000700030008000200050001000200040003000200040004000000030003000100010002000100030003000000010002000200000000000500000002000000000002000100040008000400070002000000030003000100020003000000000003000300020004000200000004000000050000000000010000000100000001000200020001000000040001000000000004000200"> : tensor<1x125xui16>
    %c_0 = stablehlo.constant dense<2> : tensor<1xui16>
    return %c, %c_0 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02000000050001000200020001000200070001000400000003000300000000000000000002000300070000000200020000000000010003000200020004000100000003000300000003000100020000000600050001000000020002000000070000000000020002000700030008000200050001000200040003000200040004000000030003000100010002000100030003000000010002000200000000000500000002000000000002000100040008000400070002000000030003000100020003000000000003000300020004000200000004000000050000000000010000000100000001000200020001000000040001000000000004000200"> : tensor<1x125xui16>
    return %c : tensor<1x125xui16>
  }
}
