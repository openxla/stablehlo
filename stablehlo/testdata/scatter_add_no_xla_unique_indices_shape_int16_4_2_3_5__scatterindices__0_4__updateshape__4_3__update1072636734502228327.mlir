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
    %0 = stablehlo.constant dense<"0x05000200FCFFFDFFFEFFFDFF0000FEFF0000FEFF0100FCFF0100FFFF0000FCFF00000000FEFFFAFF02000A000000FEFF030001000300040002000000FEFFFFFF00000100000004000400FDFFFFFFFEFF0300FEFFFEFF0600FCFF05000000010000000100000001000000FEFF0100FEFFFFFF02000000FDFF030000000300FDFF00000500FCFFFFFFFFFF020001000100030000000300FEFFFDFF00000200FDFF0600FCFF0300FEFF0200FEFFFDFF0000FDFFFEFF0200000000000000FEFF03000600FBFF0200FFFFFDFF00000200FDFF0000FCFFFFFF0300FCFFFCFFFDFFFDFF0100FCFFFAFF00000100020000000100"> : tensor<4x2x3x5xi16>
    %1 = stablehlo.constant dense<[[-4, 1, -6], [-2, 3, 4], [0, -3, 1], [-4, -3, 3]]> : tensor<4x3xi16>
    return %0, %1 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> tensor<4x2x3x5xi16> {
    %0 = stablehlo.constant dense<"0x05000200FCFFFDFFFAFFFDFF0000FEFF0000FFFF0100FCFF0100FFFFFAFFFCFF00000000FEFFFAFF02000A000000FEFF030001000300040002000000FEFFFFFF00000100FEFF04000400FDFFFFFF01000300FEFFFEFF0600000005000000010000000100000001000000FEFF0100FEFFFFFF02000000FDFF030000000300FDFF00000500FCFFFFFFFFFFFFFF01000100030000000400FEFFFDFF00000200FDFF0600FCFF0300FEFF0200FEFFFDFF0000FDFFFEFF0200000000000000FAFF03000600FBFF0200FCFFFDFF00000200FDFF0300FCFFFFFF0300FCFFFCFFFDFFFDFF0100FCFFFAFF00000100020000000100"> : tensor<4x2x3x5xi16>
    return %0 : tensor<4x2x3x5xi16>
  }
}

