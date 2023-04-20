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
    %0 = stablehlo.constant dense<"0x030000000200040001000000020000000000000002000400020002000600040000000100000000000000020002000500090003000300010000000100020004000300000006000200020002000100020002000300030007000400010002000200000004000000000002000700040001000200010004000300040000000000040003000000020003000000010002000000000002000400010004000100060001000000030002000400020001000100020001000100010001000200030003000400040001000100010003000100020000000100000000000500010000000200040000000100020001000000010002000200"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[0, 3, 1], [5, 2, 1], [0, 1, 2], [2, 3, 2]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x030000000200040000000000020000000000000002000400020002000100040000000100000000000000020002000500090003000300010000000100020004000300000005000200020002000100020002000300030007000100010002000200000004000000000002000700040001000200010004000300040000000000040000000000020003000000010002000000000002000200010004000100060001000000030002000400020001000100020001000100010001000200030002000400040001000100010003000100020000000100000000000500010000000200040000000100020001000000010002000200"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

