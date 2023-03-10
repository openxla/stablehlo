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
    %0 = stablehlo.constant dense<"0xFEFFFFFF06000200FCFF0100FEFF0100020000000300FFFF0000020000000000040001000200FFFFFDFF00000100000001000000000001000200FCFF000000000000FEFFFDFF00000200000003000500FEFFFDFF00000100FBFF0200FEFF07000000FBFFFBFFFEFFFFFF000000000200000002000200FDFF010001000000FFFF00000100030007000500000003000300FEFFFBFF03000100FCFF02000000FFFFFEFF02000000FCFF0000FDFF0200FFFF0000FCFF02000200010000000100000000000200FFFF00000100020000000000FEFF0000020000000100FFFFFEFF0000FFFF0000FFFF020004000700F8FF0300"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[-2, 0, 0], [-1, 1, 0], [-3, 6, -3], [1, 3, 0]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0xFEFFFFFF06000200FCFF0100FEFF0100020000000300FFFF0000020000000000040001000200FFFFFDFF00000100000001000000000001000200FCFF000000000000FEFFFDFF00000200000003000100FEFFFDFF00000100FBFF0200FEFF07000000FBFFFBFFFEFFFFFF000000000200000002000200FDFF010001000000FFFFFDFF0100030007000500000003000300FEFFFBFFFDFF0100FCFF02000000FFFFFEFF02000000FCFF0000FDFF0200FFFF0000FCFF02000200010000000100000000000200FFFF00000100020000000000FEFF0000020000000100FFFFFEFF0000FFFF0000FFFF020004000700F8FF0300"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

