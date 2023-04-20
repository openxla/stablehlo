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
    %0 = stablehlo.constant dense<"0x0100FAFF01000300FFFF0300000000000300FEFF0300FAFF04000600FDFFFFFFFEFF0100FFFF0400010000000300FEFF0000FFFF000001000200FFFF0100FFFFFFFF0100010001000000FDFF0100FFFF0400FEFFFEFFFFFF04000400FDFFFCFFFDFFFEFF03000000040001000100FFFFFDFF01000000FFFF000003000200FFFF0300FEFF0100FCFF0100FBFF04000300FEFFFBFF01000000FFFF0000FBFFFFFFFEFF01000300FDFF030004000000FBFF02000400FEFF000000000000FFFF010003000000FFFF020002000200FEFF0000FFFF03000100000000000000000000000000040000000300020001000000FDFF"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[0, 0, 0], [-4, 4, 0], [0, -6, 3], [-2, 6, 1]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x0100FAFF010003000000030000000000030000000300FAFF040006000000FFFFFEFF0100FFFF0400010000000300FEFF0000FFFF000001000200FFFF0100FFFFFFFF0100FCFF01000000FDFF0100FCFF0400FEFFFEFFFFFF00000400FDFFFCFFFDFFFEFF03000000040001000100FFFFFDFF01000000FFFF000003000200FFFF0000FEFF0100FCFF01001E0004000300FEFFFBFF03000000FFFF0000FBFFFFFFFEFF01000300FDFF030004000000FBFF02000400FEFF0000000000000200010003000000FFFF0C0002000200FEFF0000FFFF03000100000000000000000000000000040000000300020001000000FDFF"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

