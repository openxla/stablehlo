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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui16>, tensor<2xi32>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>) {
    %0 = stablehlo.constant dense<"0x040000000300000004000000000006000300000001000100010003000100030001000400020001000000050001000000010001000200000005000300000000000200040000000100010002000200010000000500040002000000000000000000010001000200000003000100000000000400000001000600000001000000040001000100000003000300020000000200010004000200030000000100000000000400010001000200010001000300010000000300030002000500000001000100050002000400020000000400040001000200040000000400010006000300030005000000020001000300020000000500"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[5, 1, 1], [2, 4, 0], [6, 5, 0], [4, 0, 5]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x040000000300000004000000000006000300000001000100010003000100030001000400020001000000050001000000010001000200000005000300000000000200040000000100010002000200010000000500040002000000000000000000010001000200000003000100000000000400000001000600000001000000040001000100000003000300020000000200010004000000030000000100000000000400010001000200010001000300010000000300030002000500000001000100050002000400000000000400040001000200040000000400010006000300030005000000020001000300020000000500"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

