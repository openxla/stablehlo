// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui16>, tensor<2xi32>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>) {
    %0 = stablehlo.constant dense<"0x000002000100030004000200010002000000010002000500010002000600000003000100020004000200040000000600030002000200010003000100020003000200010000000200030006000100020001000100050002000300020002000000000000000000020002000000020003000600030003000200030005000100000005000100000002000100030000000300040003000500030000000100020000000300000000000100020001000000040003000100000003000200000005000100000002000000000001000100040000000300000000000100020007000200020002000100020000000000020001000000"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[3, 3, 2], [0, 0, 1], [2, 3, 4], [1, 0, 4]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x00000200010003000C000200010002000000030002000500010002000C0000000300010002000400020004000000060003000200020001000300010002000300020001000000020003000600010000000100010005000200030002000200000000000000000002000200000002000300060003000300020003000500010000000A000100000002000100090000000300040003001400030000000100020000000300000000000100020001000000040003000100000003000200000005000100000002000000000001000100040000000C00000000000100020007000200020002000100020000000000020001000000"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

