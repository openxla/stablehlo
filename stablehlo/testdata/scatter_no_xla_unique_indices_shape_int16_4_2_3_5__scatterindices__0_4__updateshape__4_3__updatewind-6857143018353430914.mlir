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
    %0 = stablehlo.constant dense<"0x00000000FEFFFFFF0400FFFFFFFFFFFFFFFF0500FEFFFFFFFFFFFBFFFDFFFFFF0100FEFFFEFF00000000FEFF000003000100FFFFFCFF020000000000FEFF00000100FEFF060000000000040003000000FAFFFFFF0200FCFFFCFFFFFF00000100FFFF000000000000040000000000000001000500FFFF0100FCFF00000000FFFF0500FFFF0100FFFFFFFF0000FFFFFEFF00000200FCFF0300FEFF01000100FFFFFFFF02000000FDFF03000100FFFF040004000100030002000300FFFFFFFF000002000000F8FF01000200FEFF01000400FBFF06000300FDFFFBFF00000100FFFFFFFFFEFFFCFF0000FDFF000000000500"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[-1, -3, 0], [6, 0, -2], [2, -4, -2], [-5, 3, 2]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x00000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFEFFFFFFFFFFFBFF0000FFFF0100FEFFFEFF00000000FEFF000003000100FFFFFCFF020000000000FEFF00000100FEFF060000000000040003000000FAFFFFFF0200FCFFFEFFFFFF00000100FFFF000000000000040000000000000001000500FFFF0100FCFF00000000FFFF0200FFFF0100FFFFFFFFFCFFFFFFFEFF00000200FEFF0300FEFF01000100FFFFFFFF02000000FDFF03000100FFFF040004000100030002000300FFFFFBFF000002000000F8FF03000200FEFF01000400020006000300FDFFFBFF00000100FFFFFFFFFEFFFCFF0000FDFF000000000500"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

