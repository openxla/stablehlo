// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %2 = call @expected() : () -> tensor<1x50x3xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<"0x01000100FCFF0000FDFFFEFFFCFF0100FFFFFDFFFCFFFDFF00000000FFFFFEFFF6FF02000000FFFF00000100030003000000030004000000FFFFFCFF0200FBFFFEFFFEFF0300FBFF01000400000002000000FFFF01000300000001000000FEFF03000000FEFFFBFFFFFFFDFF020001000100FEFFFBFF0100FDFF0500FEFFFFFFFBFF02000100000002000300FDFF0300010000000000030002000100FCFFFDFFFCFF01000400000000000100040002000200030001000000FCFF000001000000FFFF0100010000000000FFFFFFFFFFFFFFFFFFFF0100020003000100FEFF01000400FFFF01000000FFFF0000020000000100040003000300FCFFF8FFFFFF00000000FEFF0200FFFF05000400000004000300FFFFFFFF0200FEFFFFFF000002000000FFFF000000000100FFFF"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[0, 0, 1]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0x01000100FCFF0000FDFFFEFFFCFF0100FFFFFDFFFCFFFDFF00000000FFFFFEFFF6FF02000000FFFF00000100030003000000030004000000FFFFFCFF0200FBFFFEFFFEFF0300FBFF01000400000002000000FFFF01000300000001000000FEFF03000000FEFFFBFFFFFFFDFF020001000100FEFFFBFF0100FDFF0500FEFFFFFFFBFF02000100000002000300FDFF0300010000000000030002000100FCFFFDFFFCFF01000400000000000100040002000200030001000000FCFF00000100000000000100010000000000FFFFFFFFFFFFFFFFFFFF0100020003000100FEFF01000400FFFF01000000FFFF0000020000000100040003000300FCFFF8FFFFFF00000000FEFF0200FFFF05000400000004000300FFFFFFFF0200FEFFFFFF000002000000FFFF000000000100FFFF"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

