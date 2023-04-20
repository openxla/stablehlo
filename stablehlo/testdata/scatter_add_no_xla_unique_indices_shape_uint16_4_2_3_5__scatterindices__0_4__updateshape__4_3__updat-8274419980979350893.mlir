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
    %0 = stablehlo.constant dense<"0x010001000200020004000300000004000600010001000100020000000000030000000400010001000600000000000300030003000300040001000100000003000100000007000200020000000600010003000000030006000000000000000100000000000200000000000100010000000000000000000200040001000100000006000100020001000300010000000000000005000100000000000000030003000000010002000200000005000100060001000100010003000000000005000300020000000100020002000000000003000000040000000100010002000300010003000200020001000000010006000000"> : tensor<4x2x3x5xui16>
    %1 = stablehlo.constant dense<[[1, 1, 2], [1, 0, 2], [0, 0, 4], [1, 5, 5]]> : tensor<4x3xui16>
    return %0, %1 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> tensor<4x2x3x5xui16> {
    %0 = stablehlo.constant dense<"0x010001000200020005000300000004000600020001000100020000000200030000000400010001000600000000000300030003000300040001000100000003000100000008000200020000000600010003000000030006000200000000000100000000000200000000000100010000000000000000000200040001000100000006000100020001000300010000000000000005000500000000000000030003000000010002000200000005000100060001000100010003000000000006000300020000000100070002000000000003000500040000000100010002000300010003000200020001000000010006000000"> : tensor<4x2x3x5xui16>
    return %0 : tensor<4x2x3x5xui16>
  }
}

