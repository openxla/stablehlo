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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi16>, tensor<2xi32>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>) {
    %0 = stablehlo.constant dense<"0x0100FFFF0000FCFF020000000000000002000200FEFFFFFF0100030005000000FBFF0200FEFFFEFF01000000000001000600FFFF0000010001000000FEFF0300FFFF0000FBFFFFFFFEFF0100FFFF0000FEFF0000000001000100070000000400FFFF0000FEFF0000FCFF0000020000000000FEFF00000100FEFF00000100F8FFFAFF000000000000010001000100FEFFFFFFFFFF010000000000FBFFFEFFFDFFFDFF02000400000000000200040004000000FEFFFFFFFDFFFDFF0400FEFF00000300000001000000FEFF00000000FEFF06000200F9FF010001000100FCFF0000FFFF000000000000FFFFFEFF0000FFFF"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[1, 6, 0], [-5, 0, 0], [0, 1, 5], [-3, -2, 1]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x0100FFFF0000FCFF020000000000000002000600FEFFFFFF0100030005000000FBFF0200FEFFFEFF01000000000001000600FFFF0000010001000000FEFF0300FFFF0000FBFFFFFFFEFF0100FFFF0000FEFF0000000001000100070000000400FFFF0000FEFF0000FCFF0000020000000000FEFF00000100FEFF00000100F8FF0000000000000000010001000100FEFFFFFFFFFF050000000000FBFFFEFFFDFFFDFF02000400000000000200040004000000FEFFFFFFFDFFFDFF0400FEFF00000300000001000000FEFF00000000FEFF06000200F9FF010001000100FCFF0000FFFF000000000000FFFFFEFF0000FFFF"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

