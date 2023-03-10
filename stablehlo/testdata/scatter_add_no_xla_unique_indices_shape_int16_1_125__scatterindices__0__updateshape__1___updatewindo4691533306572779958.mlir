// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %2 = call @expected() : () -> tensor<1x125xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi16>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x125xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi16>, tensor<1x125xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi16>, tensor<1xi16>) {
    %0 = stablehlo.constant dense<"0xFDFF0000F9FFFFFF000001000200FFFF00000100FEFFFBFFFEFFFFFF000004000300FFFF0200000002000100FFFF0100FAFFFFFF02000000010000000300050008000100FFFF0000FFFFFEFF06000200FFFF0100FFFF0300000004000300FCFF01000000010000000000060003000000FEFFFEFFFFFF0400FCFF0300FDFFFFFF00000200000001000500FEFFFCFF0000FAFF010000000100FEFF0200FEFFFCFFFDFF01000200FFFF00000000FDFF000000000200FDFF02000200FEFF0000FBFF000002000000FCFF01000200FEFFFEFFFFFF000004000000050000000100FFFF0000000000000200FEFF0300FFFF0300FDFF000000000000FDFF"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<1> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0xFEFF0000F9FFFFFF000001000200FFFF00000100FEFFFBFFFEFFFFFF000004000300FFFF0200000002000100FFFF0100FAFFFFFF02000000010000000300050008000100FFFF0000FFFFFEFF06000200FFFF0100FFFF0300000004000300FCFF01000000010000000000060003000000FEFFFEFFFFFF0400FCFF0300FDFFFFFF00000200000001000500FEFFFCFF0000FAFF010000000100FEFF0200FEFFFCFFFDFF01000200FFFF00000000FDFF000000000200FDFF02000200FEFF0000FBFF000002000000FCFF01000200FEFFFEFFFFFF000004000000050000000100FFFF0000000000000200FEFF0300FFFF0300FDFF000000000000FDFF"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

