// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>)
    %1 = call @expected() : () -> tensor<4x2x3x5xui16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    return %2 : tensor<4x2x3x5xui16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}, tensor<4x3xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030000000100010001000000000001000100010004000300030000000000000003000400010005000100020000000500000001000100010000000300000002000600010001000100030000000000010002000300040005000600010003000400020001000000040003000000000005000400010002000500020003000100000003000200050001000300040003000000050004000000030000000600020000000300030001000200010001000000000001000200000002000200030002000300020002000300040000000400010001000200040003000300020000000000010001000100060001000000030001000100"> : tensor<4x2x3x5xui16>
    %c_0 = stablehlo.constant dense<[[0, 1, 2], [2, 0, 0], [5, 1, 1], [2, 8, 2]]> : tensor<4x3xui16>
    return %c, %c_0 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030000000100010001000000000001000100010004000300030000000200000003000400010005000100020000000500000001000100010000000300000002000600010002000100030000000000010002000300040005000600010003000400020001000000040003000000000005000400010002000500020003000100000005000200050001000300040003000000050004000100030000000600020000000300030001000200010001000000000001000200000002000200030002000300020002000300080000000400010001000200040003000300020000000000010001000100060001000000030001000100"> : tensor<4x2x3x5xui16>
    return %c : tensor<4x2x3x5xui16>
  }
}
