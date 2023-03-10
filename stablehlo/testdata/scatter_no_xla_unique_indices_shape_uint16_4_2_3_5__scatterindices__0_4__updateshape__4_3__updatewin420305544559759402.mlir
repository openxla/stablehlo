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
      stablehlo.return %arg1 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui16>, tensor<2xi32>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>) {
    %0 = stablehlo.constant dense<"0x010000000100000004000000010005000600040001000200040002000200030000000100050000000100000001000400000000000000000002000200010002000000020000000100040003000400050000000000040002000100010000000300010003000000040002000000010006000600010000000500020002000200010000000200010001000100010003000200000001000000000003000400020003000100040002000000020000000200000000000500010002000000000002000000010002000000000005000300020002000400000002000000010002000400030000000400020003000100030003000000"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[1, 4, 3], [2, 4, 4], [3, 6, 0], [2, 3, 1]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x010000000100000001000000010005000600040001000200040002000300030000000100050000000100000001000400000000000000000002000200010002000000020002000100040003000400040000000000040002000400010000000300010003000000040002000000010006000600010000000500020002000200010003000200010001000100060003000200000001000000000003000400020003000100040002000000020000000200000000000500010002000000000002000000010002000000030005000300020002000100000002000000010002000400030000000400020003000100030003000000"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

