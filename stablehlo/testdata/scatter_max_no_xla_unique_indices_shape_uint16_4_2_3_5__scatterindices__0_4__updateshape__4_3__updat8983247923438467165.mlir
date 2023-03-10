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
    %0 = stablehlo.constant dense<"0x030001000300000001000200030002000000010000000200010004000300040001000400040001000100030001000200000000000100020000000100000002000000000001000200000001000200000000000100030003000100010000000200010003000100000002000000040002000700040002000100040000000500030001000100060003000300050005000200000002000400050001000100020001000200040001000000070002000100010001000100020001000100000000000200000007000600000003000200020008000200040001000500000000000000010000000000010001000100010000000300"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[0, 2, 0], [5, 1, 2], [2, 1, 4], [4, 1, 3]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x030001000300000001000200030002000000020000000200010004000300040001000400040001000100030001000200000000000100020000000100000002000000000005000200000001000200010000000100030003000200010000000200010003000100000002000000040002000700040002000100040000000500030002000100060003000300050005000200000002000400050001000100020001000200040001000000070002000100010001000100020001000100000004000200000007000600010003000200020008000300040001000500000000000000010000000000010001000100010000000300"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

