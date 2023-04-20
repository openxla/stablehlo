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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xui16>, tensor<1xi32>, tensor<1xui16>) -> tensor<1x125xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xui16>, tensor<1x125xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xui16>, tensor<1xui16>) {
    %0 = stablehlo.constant dense<"0x01000000030001000100040004000100020001000400010006000400020000000000000000000500000001000400060001000300030006000000020000000000020000000500040004000100010002000100010005000100010007000000030001000000000000000100020001000100010000000200000000000400080009000200000002000100020000000000010001000000010002000100040003000000030000000000040000000300030000000100060003000100030000000000030001000400010000000000000004000100050002000200020002000200000001000000040003000000010001000100060001000100080002000200"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<2> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x01000000030001000100040004000100020001000400010006000400020000000000000000000500000001000400060001000300030006000000020000000000020000000500040004000100010002000100010005000100010007000000030001000000000000000100020001000100010000000200000000000400080009000200000002000100020000000000010001000000010002000100040003000000030000000000040000000300030000000100060003000100030000000000030001000400010000000000000004000100050002000200020002000200000001000000040003000000010001000100060001000100080002000200"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

