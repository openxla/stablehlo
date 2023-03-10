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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xui16>, tensor<1xi32>, tensor<1xui16>) -> tensor<1x125xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xui16>, tensor<1x125xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xui16>, tensor<1xui16>) {
    %0 = stablehlo.constant dense<"0x01000200020003000000010002000200050000000400030004000000030004000400010001000000070002000100010001000100020000000000050001000500000003000100000002000200030000000100010003000300000001000600040004000200010004000100050004000000000000000000040001000000020000000700020000000100000004000200000001000400060002000300000001000000010001000000010000000400040000000300070003000000040000000200000004000400010000000200020006000200040001000000010003000500050001000200020000000100000000000100000003000300010002000200"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<3> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x03000200020003000000010002000200050000000400030004000000030004000400010001000000070002000100010001000100020000000000050001000500000003000100000002000200030000000100010003000300000001000600040004000200010004000100050004000000000000000000040001000000020000000700020000000100000004000200000001000400060002000300000001000000010001000000010000000400040000000300070003000000040000000200000004000400010000000200020006000200040001000000010003000500050001000200020000000100000000000100000003000300010002000200"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

