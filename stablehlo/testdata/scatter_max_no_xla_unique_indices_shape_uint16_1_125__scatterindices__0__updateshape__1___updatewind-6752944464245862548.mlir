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
    %0 = stablehlo.constant dense<"0x00000100050002000000000000000200020000000100080000000000020000000400000003000400020003000000020002000300000000000200060003000000060001000100020002000400020003000400000001000100010001000100010003000400010000000000030003000100010001000000010002000100000001000000080004000200040001000100010004000000030003000200010000000100020005000200010002000200030005000100010002000000000004000000000003000100040005000200010002000000050002000300010002000300000003000100000001000200010000000000010003000000010000000100"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<1> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x01000100050002000000000000000200020000000100080000000000020000000400000003000400020003000000020002000300000000000200060003000000060001000100020002000400020003000400000001000100010001000100010003000400010000000000030003000100010001000000010002000100000001000000080004000200040001000100010004000000030003000200010000000100020005000200010002000200030005000100010002000000000004000000000003000100040005000200010002000000050002000300010002000300000003000100000001000200010000000000010003000000010000000100"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

