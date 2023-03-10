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
    %0 = stablehlo.constant dense<"0x00000100030002000400030002000200000000000100000001000400030005000100010000000200020001000300030002000300020002000300010000000500030002000100020000000000020004000100030000000200030003000500030001000100050001000100030001000300000000000000000000000100030001000300020001000700050000000500000000000000040000000000000003000400030002000100030000000100050005000000010002000000030004000300010000000000070002000400030002000400030000000300000001000000060003000000020003000500000002000100000002000000000000000000"> : tensor<1x125xui16>
    %1 = stablehlo.constant dense<3> : tensor<1xui16>
    return %0, %1 : tensor<1x125xui16>, tensor<1xui16>
  }
  func.func private @expected() -> tensor<1x125xui16> {
    %0 = stablehlo.constant dense<"0x03000100030002000400030002000200000000000100000001000400030005000100010000000200020001000300030002000300020002000300010000000500030002000100020000000000020004000100030000000200030003000500030001000100050001000100030001000300000000000000000000000100030001000300020001000700050000000500000000000000040000000000000003000400030002000100030000000100050005000000010002000000030004000300010000000000070002000400030002000400030000000300000001000000060003000000020003000500000002000100000002000000000000000000"> : tensor<1x125xui16>
    return %0 : tensor<1x125xui16>
  }
}

