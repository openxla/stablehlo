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
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xui16>, tensor<1xi32>, tensor<1xui16>) -> tensor<1x125xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xui16>, tensor<1x125xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xui16>, tensor<1xui16>) {
    %0 = stablehlo.constant dense<"0x01000100000001000000040002000700070006000000000000000100010000000100060003000000030004000200020000000500000001000500040003000100000006000500010001000100040000000000010002000000020003000200030004000400000003000000020004000200020002000400060003000000000001000000000000000200030000000300020002000500000002000200000002000600020004000600020000000600030004000200030001000700020003000300030003000000020003000400010000000000020000000000060002000100040002000000020000000200010002000500000006000000020005000000"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<0> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x01000100000001000000040002000700070006000000000000000100010000000100060003000000030004000200020000000500000001000500040003000100000006000500010001000100040000000000010002000000020003000200030004000400000003000000020004000200020002000400060003000000000001000000000000000200030000000300020002000500000002000200000002000600020004000600020000000600030004000200030001000700020003000300030003000000020003000400010000000000020000000000060002000100040002000000020000000200010002000500000006000000020005000000"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

