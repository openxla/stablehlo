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
    %0 = stablehlo.constant dense<"0x02000000060000000400070002000500040000000100010001000000010000000100020000000100000001000100020006000000030004000200050002000800000000000100000003000500040001000400000002000100010002000000000000000100020001000100000007000000000000000000020006000400010004000000010000000100040004000100000001000200020001000200040000000000010003000300020002000400010000000200020003000500000000000600000000000000010000000000010003000400060004000100010000000100040004000300000001000100040005000300000003000400050003000000"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<2> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x02000000060000000400070002000500040000000100010001000000010000000100020000000100000001000100020006000000030004000200050002000800000000000100000003000500040001000400000002000100010002000000000000000100020001000100000007000000000000000000020006000400010004000000010000000100040004000100000001000200020001000200040000000000010003000300020002000400010000000200020003000500000000000600000000000000010000000000010003000400060004000100010000000100040004000300000001000100040005000300000003000400050003000000"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

