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
      %5 = stablehlo.add %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<"0x0000020000000300FEFF0000FFFF0100FEFF0000FFFFFEFF000001000000FFFF0000010003000000FFFFFEFFFDFF03000300FDFFFEFF0200000000000300FCFF00000200FFFFFEFF00000000FFFFFFFF000000000300FFFF0400FDFF0000FEFF0200FFFF0000FEFF0000FDFF050000000000020006000000FCFF0000FFFFFEFFFEFF0000FBFFFEFFFEFF03000000FCFF01000300020000000100010001000000020001000000010000000000FDFF000000000000010002000000FEFFFFFF0100FEFFFEFF00000100FFFF0100FFFF000000000300010005000000FFFF02000000FAFF00000000FEFF0100060000000300000000000300F8FF0200FFFF0200000001000000FFFF000001000000FBFF000000000500FEFFFFFF0100FFFF0100040004000400FFFF020000000200"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[1, 2, 4]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0x0000020000000300FEFF0000FFFF0100FEFF0000FFFFFEFF000001000000FFFF0000010003000000FFFFFEFFFDFF03000300FDFFFEFF0200000000000300FCFF00000200FFFFFEFF00000000FFFFFFFF000000000300FFFF0400FDFF0000FEFF0200FFFF0000FEFF0000FDFF050000000000020006000000FCFF0000FFFFFEFFFEFF0000FBFFFEFFFEFF03000000FCFF01000300020000000100010001000000020001000000010000000000FDFF000000000000010002000000FEFFFFFF0100FFFF000004000100FFFF0100FFFF000000000300010005000000FFFF02000000FAFF00000000FEFF0100060000000300000000000300F8FF0200FFFF0200000001000000FFFF000001000000FBFF000000000500FEFFFFFF0100FFFF0100040004000400FFFF020000000200"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

