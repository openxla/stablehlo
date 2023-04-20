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
      stablehlo.return %arg1 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<"0x02000000FBFF0000000000000000FFFFFEFFFDFFFEFFFDFFFDFF05000000FFFF0500FEFF000000000000FFFFFFFF0300010000000000FEFFFDFF0200FEFF0100FBFFFBFFFFFF0300FEFF01000300000006000100FEFFFDFF0200FEFF0000FCFFFDFF0100FEFFFBFF0200FDFFFFFF0400FCFF01000000FEFF0000FEFFFEFF0000FFFF0500000004000500FFFF04000200FBFF01000200000003000100000004000200000000000200FBFFFFFF00000100050002000200FAFFFFFF0500FEFFFFFFFFFF0400010000000000000000000300000000000000060000000000000000000000000000000600FFFF00000000000004000000FFFFFEFF00000400000000000000FEFF05000500FFFFFFFFFCFF0400000003000200030005000700FFFF0000FCFFFDFFFAFF02000000FFFF"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[3, -2, -1]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0x02000000FBFF0000000000000000FFFFFEFFFDFFFEFFFDFFFDFF05000000FFFF0500FEFF000000000000FFFFFFFF0300010000000000FEFFFDFF0200FEFF0100FBFFFBFFFFFF0300FEFF01000300000006000100FEFFFDFF0200FEFF0000FCFFFDFF0100FEFFFBFF0200FDFFFFFF0400FCFF01000000FEFF0000FEFFFEFF0000FFFF0500000004000500FFFF04000200FBFF01000200000003000100000004000200000000000200FBFFFFFF00000100050002000200FAFFFFFF0500FEFFFFFF0300FEFFFFFF00000000000000000300000000000000060000000000000000000000000000000600FFFF00000000000004000000FFFFFEFF00000400000000000000FEFF05000500FFFFFFFFFCFF0400000003000200030005000700FFFF0000FCFFFDFFFAFF02000000FFFF"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

