// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xui16>, tensor<1xui16>)
    %2 = call @expected() : () -> tensor<1x125xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xui16>, tensor<1xi32>, tensor<1xui16>) -> tensor<1x125xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xui16>, tensor<1x125xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xui16>, tensor<1xui16>) {
    %0 = stablehlo.constant dense<"0x00000000030003000400010003000000020004000500010001000500000002000400000006000300010006000200000005000100030002000200040003000100010004000800010006000000030001000000010001000200020000000700020004000000000000000000000002000000020000000300010007000200010005000400040000000300000005000200020006000100030000000200010001000000000000000300050005000500020001000600030001000600010003000300010002000100050000000300040002000200020003000200010000000300030002000200030002000500000002000100000000000100010001000200"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<3> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x03000000030003000400010003000000020004000500010001000500000002000400000006000300010006000200000005000100030002000200040003000100010004000800010006000000030001000000010001000200020000000700020004000000000000000000000002000000020000000300010007000200010005000400040000000300000005000200020006000100030000000200010001000000000000000300050005000500020001000600030001000600010003000300010002000100050000000300040002000200020003000200010000000300030002000200030002000500000002000100000000000100010001000200"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

