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
    %0 = stablehlo.constant dense<"0x050004000000040000000200020000000300010001000400020007000000030000000200010001000000030005000300010001000200040001000000000000000200030001000200020001000000020002000400040001000000050008000200020000000000030003000700010003000100030000000000010004000300030000000300060004000000020005000300000006000000050001000200020000000100040004000400030000000000020002000300000001000100020001000100010005000000000002000200040001000200010001000000010004000100000005000000010003000100000000000000"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[0, 1, 3], [5, 0, 3], [1, 1, 4], [4, 4, 2]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x050004000000040000000200020000000300010001000400020007000300030000000200010001000000030005000300010001000200040001000000000000000200030005000200020001000000000002000400040001000300050008000200020000000000030003000700010003000100030000000000010004000300030001000300060004000000010005000300000006000400050001000200020000000100040004000400030000000000020002000300000001000100020004000100010005000000040002000200040001000200010001000000010004000100000005000000010003000100000000000000"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

