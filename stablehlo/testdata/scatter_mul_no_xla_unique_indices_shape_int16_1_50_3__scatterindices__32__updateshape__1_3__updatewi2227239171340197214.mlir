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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<"0xFFFFFFFFFFFF02000000FFFFFBFFFFFFFFFFF9FFFCFFFBFF0000000001000400FFFFFFFFFFFF0000020000000100FAFFF8FFFFFF0000FFFF00000200FEFF0600FDFF0400FFFF0000FFFFFEFF0200000001000400000000000500FDFFFEFFFDFF00000100FAFFFCFFFBFF0500FDFFFDFFFDFF0200FDFFFBFF03000000F9FFFFFFFFFF0000FEFFFDFF0000FAFF00000500FEFFFFFF00000000FEFFFFFF0000FFFFFFFFFCFF00000000FDFFFEFF01000300FEFFFFFF02000000FFFF0300FDFF01000400FEFFFDFF02000000FFFF020000000100FDFFFCFF00000000040006000300FEFF0100FDFF02000000FEFF0000FDFF00000300FBFF0000FEFF010000000000FEFF02000100050004000200FFFF02000100FFFF010004000200FEFF0000FBFFFCFFFEFFFFFF05000000FCFF"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[-1, 5, 2]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0xFFFFFFFFFFFF02000000FFFFFBFFFFFFFFFFF9FFFCFFFBFF0000000001000400FFFFFFFFFFFF0000020000000100FAFFF8FFFFFF0000FFFF00000200FEFF0600FDFF0400FFFF0000FFFFFEFF0200000001000400000000000500FDFFFEFFFDFF00000100FAFFFCFFFBFF0500FDFFFDFFFDFF0200FDFFFBFF03000000F9FFFFFFFFFF0000FEFFFDFF0000FAFF00000500FEFFFFFF00000000FEFFFFFF0000FFFFFFFFFCFF00000000FDFFFEFF01000300FEFFFFFF02000000FFFF0300FDFF0100FCFFF6FFFAFF02000000FFFF020000000100FDFFFCFF00000000040006000300FEFF0100FDFF02000000FEFF0000FDFF00000300FBFF0000FEFF010000000000FEFF02000100050004000200FFFF02000100FFFF010004000200FEFF0000FBFFFCFFFEFFFFFF05000000FCFF"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

