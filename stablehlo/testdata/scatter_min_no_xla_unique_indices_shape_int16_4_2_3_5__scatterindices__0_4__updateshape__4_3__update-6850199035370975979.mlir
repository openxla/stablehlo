// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi16>, tensor<2xi32>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>) {
    %0 = stablehlo.constant dense<"0x00000400FFFF0700030002000600FFFF0000020001000100FEFF0000FFFF0000FAFFFEFF020000000100FFFF0400040000000100FFFFFFFF0000FDFFFFFF0000FEFF0000FCFFFCFFFAFFFBFF0100FDFF03000000010004000600FFFF0500FFFFFAFFFFFF000000000000FFFF01000200020002000000FDFFFFFF00000200FFFFFCFF0000FEFFFDFF0200FEFFFEFF0000FDFF02000100FFFFF9FFFCFF0000FFFFFFFFFEFF01000000FBFF000001000400FDFFFDFFFDFFFDFF0000000000000300FEFFFFFF0000F9FF01000000FFFF000000000000FCFF020001000100000000000100FFFFFEFF0400FFFFFEFFFDFFFDFF"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[-3, -4, 3], [-5, 0, -2], [2, 6, -7], [1, -2, 0]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x00000400FFFF0700FDFF02000600FFFF0000FCFF01000100FEFF0000FFFF0000FAFFFEFF020000000100FFFF0400040000000100FFFFFFFF0000FDFFFFFF0000FEFF0000FBFFFCFFFAFFFBFF0100FDFF0300000001000400FEFFFFFF0500FFFFFAFFFFFF000000000000FFFF01000200020002000000FDFFFFFF00000200FFFFFCFF0000FEFFFDFF0200FEFFFEFF0000FDFF0200F9FFFFFFF9FFFCFF0000FFFFFFFFFEFF01000000FBFF000001000400FDFFFDFFFDFFFDFF0000000000000300FEFFFFFF0000F9FF01000000FFFF000000000000FCFF020001000100000000000100FFFFFEFF0400FFFFFEFFFDFFFDFF"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

