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
      stablehlo.return %arg1 : tensor<ui16>
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    return %2 : tensor<1x125xui16>
  }
  func.func private @inputs() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}, tensor<1xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00000300040005000000030001000300030001000200050003000700020001000200020002000300020002000200040002000400030001000000050002000600000002000100020001000600000005000200040001000200000001000200040002000500050000000100020002000000010000000500040003000400050001000400000005000200030001000600000002000000010000000100000000000100050000000300010003000100010002000500040002000200010000000200040000000300000000000100010001000200040000000400030000000000060001000500000004000100050000000200020000000100030001000100"> : tensor<1x125xui16>
    %c_0 = stablehlo.constant dense<2> : tensor<1xui16>
    return %c, %c_0 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02000300040005000000030001000300030001000200050003000700020001000200020002000300020002000200040002000400030001000000050002000600000002000100020001000600000005000200040001000200000001000200040002000500050000000100020002000000010000000500040003000400050001000400000005000200030001000600000002000000010000000100000000000100050000000300010003000100010002000500040002000200010000000200040000000300000000000100010001000200040000000400030000000000060001000500000004000100050000000200020000000100030001000100"> : tensor<1x125xui16>
    return %c : tensor<1x125xui16>
  }
}
