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
      stablehlo.return %arg1 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi16>, tensor<2xi32>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>) {
    %0 = stablehlo.constant dense<"0x060001000300FDFF0200060000000000FEFF0200000000000200FEFF01000300FEFF0400070000000100FAFFFCFFFEFFFEFFFCFFFAFFFFFF0200FEFF0100FEFF0100FCFF000005000200F8FF02000000FEFF0100FDFFFEFF01000300030000000000FFFF03000100030002000100FFFFFEFFFFFF030001000200FFFF01000000FFFFFDFFFEFF0000010000000000FBFF00000000030001000000FDFFFBFF0200FCFF020000000000FDFF02000000000001000100FFFF000000000000FEFF0000FFFFFFFF0200FFFF0000F9FFFCFFFFFF0200FBFF00000000070004000600FEFFFCFF0000FFFFFFFF0000000000000000"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[-2, 1, 0], [1, 3, -4], [1, -4, 0], [0, 1, -1]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x060001000300FDFFFEFF060000000000FEFF0100000000000200FEFF00000300FEFF0400070000000100FAFFFCFFFEFFFEFFFCFFFAFFFFFF0200FEFF0100FEFF0100FCFF010005000200F8FF02000300FEFF0100FDFFFEFFFCFF0300030000000000FFFF03000100030002000100FFFFFEFFFFFF030001000200FFFF010000000100FDFFFEFF00000100FCFF0000FBFF00000000000001000000FDFFFBFF0200FCFF020000000000FDFF02000000000001000100FFFF00000000000000000000FFFFFFFF020001000000F9FFFCFFFFFFFFFFFBFF00000000070004000600FEFFFCFF0000FFFFFFFF0000000000000000"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

