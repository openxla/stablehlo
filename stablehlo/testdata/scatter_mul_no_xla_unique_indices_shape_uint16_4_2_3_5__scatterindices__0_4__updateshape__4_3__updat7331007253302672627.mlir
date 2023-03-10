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
    %0 = stablehlo.constant dense<"0x000001000400000003000200010001000300010000000100020002000100000002000000020000000100010000000000030000000300020003000200010002000500010002000000030002000100000003000100010003000200030002000200020003000200030001000200010003000100010000000000020000000300050001000300020001000300020003000000010003000100000004000300020003000400040002000300010003000300000004000000000002000300050006000200010001000000000001000100020002000000020002000400000000000100020004000300010000000300000005000500"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[3, 1, 4], [4, 4, 4], [1, 5, 3], [2, 4, 0]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x0000010004000000090002000100010003000100000001000200020004000000020000000200000001000100000000000300000003000200030002000100020005000100080000000300020001000000030001000100030008000300020002000200030002000300010002000100030001000100000000000200000003000500010003000200010003000A000300000001000300030000000400030002000300040004000200030001000300030000000400000000000200030005000C000200010001000000000001000100020002000000020002000400000000000100020004000300010000000300000005000500"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

