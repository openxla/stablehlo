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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui16>, tensor<2xi32>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>) {
    %0 = stablehlo.constant dense<"0x000002000200020002000600020003000400000000000100000005000000020001000100010007000000020000000100030000000000010001000000010003000100040001000200000001000200010003000200030003000000020004000200000004000000010002000000020000000300010002000100000005000300000000000100010002000000000001000000020000000000000002000700040001000100000004000100030001000100010007000600000003000000040000000000000005000000020007000000030003000300080000000400020001000100020008000600040001000200010000000300"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[3, 3, 3], [0, 1, 2], [3, 5, 1], [5, 0, 0]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x000002000200020003000600020003000400030000000100000005000300020001000100010007000000020000000100030000000000010001000000010003000100040001000200000001000200010003000200030003000200020004000200000004000000010002000000020000000300010002000100000005000300000003000100010002000000050001000000020000000100000002000700040001000100000004000100030001000100010007000600000003000000040005000000000005000000020007000000030003000300080000000400020001000100020008000600040001000200010000000300"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

