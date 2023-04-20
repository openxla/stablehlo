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
    %0 = stablehlo.constant dense<"0x01000400040000000100010004000200050003000300000000000300010002000200030004000000040001000700030001000100060003000100000001000000010001000200000003000100030000000500010001000100010003000600050002000200020002000400000000000600020000000000040003000100000000000500000002000100010002000000010001000200000003000400040001000300020000000300010002000300080001000100020004000000000003000300040000000300020001000200010000000200000004000000020003000100030000000000010003000100000001000100010001000400000001000400"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<2> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x02000400040000000100010004000200050003000300000000000300010002000200030004000000040001000700030001000100060003000100000001000000010001000200000003000100030000000500010001000100010003000600050002000200020002000400000000000600020000000000040003000100000000000500000002000100010002000000010001000200000003000400040001000300020000000300010002000300080001000100020004000000000003000300040000000300020001000200010000000200000004000000020003000100030000000000010003000100000001000100010001000400000001000400"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

