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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    return %2 : tensor<4x2x3x5xui16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}, tensor<4x3xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x04000200010000000300020003000300010006000200050000000000010004000100000005000300010001000100050004000300000003000300000005000700020004000100020001000200010002000700020001000500010002000200000002000300020001000100010003000000010003000000040000000000020002000000000000000300050005000200030002000100020001000400020001000000060003000100010002000200020000000A000200040001000400040002000200010000000300010001000100050004000300050003000300050000000000010003000100010003000100060003000000"> : tensor<4x2x3x5xui16>
    %c_0 = stablehlo.constant dense<[[2, 0, 2], [2, 6, 4], [1, 1, 1], [1, 1, 6]]> : tensor<4x3xui16>
    return %c, %c_0 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0400020001000000060002000300030001000000020005000000000002000400010000000500030001000100010005000400030000000300030000000500070002000400020002000100020001000C000700020001000500040002000200000002000300020001000100010003000000010003000000040000000000020002000000000000000300050005000200030002000100020001000400020001000000060003000100010002000200020000000A000200040001000400040002000200010000000300010001000100050004001200050003000300050000000000010003000100010003000100060003000000"> : tensor<4x2x3x5xui16>
    return %c : tensor<4x2x3x5xui16>
  }
}
