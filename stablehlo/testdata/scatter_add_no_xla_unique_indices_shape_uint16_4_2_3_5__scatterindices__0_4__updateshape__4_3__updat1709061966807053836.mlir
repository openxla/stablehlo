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
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui16>, tensor<2xi32>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>) {
    %0 = stablehlo.constant dense<"0x050006000100010001000000000004000300010005000400010000000000020004000400030005000100000002000000020002000800000004000200000001000200010005000500000001000200060004000500000003000100030002000200030001000400000002000300010000000000020004000400020005000100010003000200010005000400030003000000010001000100040005000000000002000100000000000300010002000200000001000000030003000100030000000100010001000400030000000000010000000000000000000200010004000200020004000500040001000500010003000200"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[4, 2, 1], [2, 4, 4], [3, 4, 3], [5, 0, 3]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x0500060001000100050000000000040003000300050004000100000001000200040004000300050001000000020000000200020008000000040002000000010002000100070005000000010002000A0004000500000003000500030002000200030001000400000002000300010000000000020004000400020005000100010006000200010005000400070003000000010001000400040005000000000002000100000000000300010002000200000001000000030003000100030005000100010001000400030000000000010000000300000000000200010004000200020004000500040001000500010003000200"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

