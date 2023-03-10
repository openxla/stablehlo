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
    %0 = stablehlo.constant dense<"0x02000100070001000400010000000A00010000000000020001000000000001000100040003000000020005000000010001000000000004000000000001000100000001000300040003000000030002000000010000000300010000000100020003000000000000000300010001000200030001000500000005000400040000000200010000000000030002000200000000000200030000000500060004000200010003000000050001000400000000000100010002000200010004000400030003000100020004000000010000000600010001000100050000000400000001000000010003000100050002000100000003000100010005000000"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<1> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x01000100070001000400010000000A00010000000000020001000000000001000100040003000000020005000000010001000000000004000000000001000100000001000300040003000000030002000000010000000300010000000100020003000000000000000300010001000200030001000500000005000400040000000200010000000000030002000200000000000200030000000500060004000200010003000000050001000400000000000100010002000200010004000400030003000100020004000000010000000600010001000100050000000400000001000000010003000100050002000100000003000100010005000000"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

