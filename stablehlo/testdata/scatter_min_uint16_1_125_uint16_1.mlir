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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<1x125xui16>, tensor<1xi64>, tensor<1xui16>) -> tensor<1x125xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xui16>, tensor<1x125xui16>) -> ()
    return %2 : tensor<1x125xui16>
  }
  func.func private @inputs() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}, tensor<1xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x04000100010003000200020003000000030000000000010001000500020002000200040000000100040003000000050002000300020000000200020005000100010000000000020001000100020001000200030004000400000003000800000002000300000003000500000002000300000002000700040000000100030002000000000001000100010001000300010006000200000001000100000001000200000000000000000003000000040000000100030003000000010000000100040006000200040005000100030002000500030002000400020001000000060009000000000004000100060000000000010005000100000001000000"> : tensor<1x125xui16>
    %c_0 = stablehlo.constant dense<0> : tensor<1xui16>
    return %c, %c_0 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> (tensor<1x125xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00000100010003000200020003000000030000000000010001000500020002000200040000000100040003000000050002000300020000000200020005000100010000000000020001000100020001000200030004000400000003000800000002000300000003000500000002000300000002000700040000000100030002000000000001000100010001000300010006000200000001000100000001000200000000000000000003000000040000000100030003000000010000000100040006000200040005000100030002000500030002000400020001000000060009000000000004000100060000000000010005000100000001000000"> : tensor<1x125xui16>
    return %c : tensor<1x125xui16>
  }
}
