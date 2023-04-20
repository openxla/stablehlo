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
      stablehlo.return %arg1 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xui16>, tensor<1xi32>, tensor<1xui16>) -> tensor<1x125xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xui16>, tensor<1x125xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xui16>, tensor<1xui16>) {
    %0 = stablehlo.constant dense<"0x040000000400010000000000000000000100040000000200000002000000030000000100030007000100000003000000010000000200020000000200030003000000010005000100030000000200040003000300010000000300020000000000030002000000040004000200040004000300000002000200010000000000010005000300010000000000060000000300030000000500010002000300010004000200000001000000020002000000030001000200050004000300050006000000000003000000010001000100010004000100050006000100000000000400030002000500030004000000020001000A0001000100040001000300"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<3> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x030000000400010000000000000000000100040000000200000002000000030000000100030007000100000003000000010000000200020000000200030003000000010005000100030000000200040003000300010000000300020000000000030002000000040004000200040004000300000002000200010000000000010005000300010000000000060000000300030000000500010002000300010004000200000001000000020002000000030001000200050004000300050006000000000003000000010001000100010004000100050006000100000000000400030002000500030004000000020001000A0001000100040001000300"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

