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
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    return %2 : tensor<4x2x3x5xui16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}, tensor<4x3xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x040002000000020002000200020000000100010003000300040000000100010004000200010001000200010003000100050003000200020004000000010004000100000001000100000002000600070003000100010002000200010004000100030003000200010002000000020002000500010000000400010001000000030000000000030001000100000000000200030000000200030001000300020003000300030004000300040001000000050003000000020003000300010002000300030004000000000005000100040000000200000001000000000000000400040002000200000000000200000004000200"> : tensor<4x2x3x5xui16>
    %c_0 = stablehlo.constant dense<[[2, 5, 0], [2, 3, 3], [1, 1, 0], [2, 3, 2]]> : tensor<4x3xui16>
    return %c, %c_0 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0400020000000200040002000200000001000600030003000400000001000100040002000100010002000100030001000500030002000200040000000100040001000000030001000000020006000A0003000100010002000500010004000100030003000200010002000000020002000500010000000400010001000000030001000000030001000100010000000200030000000200030001000300020003000300030004000300040001000000050003000000020003000300010004000300030004000000030005000100040000000400000001000000000000000400040002000200000000000200000004000200"> : tensor<4x2x3x5xui16>
    return %c : tensor<4x2x3x5xui16>
  }
}
