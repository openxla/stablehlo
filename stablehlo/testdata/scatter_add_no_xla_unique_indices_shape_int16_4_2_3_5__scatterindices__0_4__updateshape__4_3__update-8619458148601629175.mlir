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
      %5 = stablehlo.add %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi16>, tensor<2xi32>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>) {
    %0 = stablehlo.constant dense<"0xFEFF000003000100FBFFFCFFFAFF00000000FDFF0100FFFF00000400FEFF00000000030000000300FBFFFEFF0100FFFF01000400FDFF000000000000000006000200FFFF0000FFFFFEFF0500FEFF05000000FFFF000003000200FFFFFFFF000004000200000001000200FFFF02000000FFFFFEFFFFFF02000000040004000000070001000000010002000000FFFFFDFF0100020001000500FFFF0000FDFF00000400FEFFFDFFFDFFFDFFFEFF0000FEFF00000200030002000000000000000200FEFF02000000FFFF000003000000FEFF000000000200FEFFFFFF02000200FFFF02000300FFFF02000300FFFFFDFF0000"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[-3, 3, -3], [0, 1, 0], [0, -5, 0], [0, 3, 6]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0xFEFF000003000100F8FFFCFFFAFF0000000000000100FFFF00000400FBFF00000000030000000300FBFFFEFF0100FFFF01000400FDFF000000000000000006000200FFFF0000FFFFFEFF0500FEFF06000000FFFF000003000200FFFFFFFF000004000200000001000200FFFF02000000FFFFFEFFFFFF0200000004000400000007000100000001000200FBFFFFFFFDFF0100020001000500FFFF0000FDFF00000400FEFFFDFFFDFFFDFFFEFF0000FEFF00000200030002000000000000000200FEFF020000000200000003000000FEFF060000000200FEFFFFFF02000200FFFF02000300FFFF02000300FFFFFDFF0000"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

