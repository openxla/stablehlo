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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi16>, tensor<2xi32>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>) {
    %0 = stablehlo.constant dense<"0x0000000000000000FEFFFAFF0200FBFF05000100FCFFFFFF000001000400FFFFFEFF00000200030001000000FFFF0300FEFFFDFF0000000001000000FAFFFFFF03000300000001000000000000000000FEFFFDFF0100030000000000FEFFFDFF040003000000FCFF0100FFFFFDFF0000FFFF000002000000FDFF00000000000000000100FFFF03000000FFFFFEFF0000FFFF0100FFFFFDFFFCFFFEFF0000020000000200FFFF01000000FFFF0000FFFF01000000FCFF0100020004000300FAFF02000100FEFFFEFF030000000100FEFF0000FFFF0100FDFF000004000000FFFF020000000100FCFFFEFFFFFF0600FFFF"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[0, -2, 2], [6, 5, -1], [-5, -2, -1], [-3, 1, -3]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x00000000000000000000FAFF0200FBFF0500FEFFFCFFFFFF000001000800FFFFFEFF00000200030001000000FFFF0300FEFFFDFF0000000001000000FAFFFFFF03000300000001000000000000000000FEFFFDFF0100030000000000FEFFFDFF040003000000FCFF0100FFFFFDFF0000FFFF000002000000FDFF00000000000000000100FFFF030000000200FEFF0000FFFF01000100FDFFFCFFFEFF0000020000000200FFFF01000000FFFF0000FFFF01000000FCFF010002000400F7FFFAFF02000100FEFFFEFF030000000100FEFF0000FFFF0100FDFF000004000000FFFF020000000100FCFFFEFFFFFF0600FFFF"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

