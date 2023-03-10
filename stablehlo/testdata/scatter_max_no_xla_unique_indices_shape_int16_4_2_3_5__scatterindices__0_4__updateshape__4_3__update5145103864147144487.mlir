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
    %0 = stablehlo.constant dense<"0x0000FFFFFEFF01000000010000000100FEFF070003000000FCFF0300FEFF0200FDFFFDFF0200FCFFFFFFFEFF00000500FBFF00000000000000000100FEFF000002000100020000000200FFFF0200FEFFFBFF02000400FEFF0200FCFF010001000100030000000200FEFF000001000000FFFF000007000000030000000300FEFF04000300010000000100FCFF03000300010001000200FCFFFFFF0600FFFFFAFF000005000100FEFFFDFF000003000100FBFFFDFF000000000000FFFF00000500FFFFFDFF0400000005000400FBFF0200FDFF02000000FAFFFAFF0500FFFF00000000FFFF0100FFFFFDFFF9FF05000300"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[-3, 0, 4], [3, 0, 0], [-3, -3, -2], [0, 0, -3]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x0000FFFFFEFF01000000010000000100FEFF070003000000FCFF030004000200FDFFFDFF0200FCFFFFFFFEFF00000500FBFF00000000000000000100FEFF000002000100030000000200FFFF02000000FBFF02000400FEFF0200FCFF010001000100030000000200FEFF000001000000FFFF000007000000030000000300FEFF04000300010000000100FDFF03000300010001000200FCFFFFFF0600FFFFFAFF000005000100FEFFFDFF000003000100FBFFFDFF000000000000FFFF00000500FFFFFDFF0400000005000400FBFF0200FDFF02000000FAFFFAFF0500FFFF00000000FFFF0100FFFFFDFFF9FF05000300"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

