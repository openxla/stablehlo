// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %2 = call @expected() : () -> tensor<1x50x3xi16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      stablehlo.return %arg1 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<"0x00000200FDFFFDFF010001000000FDFF0400FDFF00000100FBFF000005000000FFFF0000FEFFFAFFFFFF00000300FEFF0000FEFFFEFF0300FEFF0300FCFF0000FEFFFDFF05000200FEFF0000040000000000FBFFFFFFFFFF000002000000FDFF02000500FEFFFEFFFCFFFEFF03000300FEFF02000000FDFFFFFF04000100FCFFFFFFFEFFFBFF020000000100000007000000FCFFFDFF0000FBFFFDFF05000200FDFF00000000FFFF0100000000000000030002000100FEFF0000F7FF0100FDFF0500FFFF03000100FEFFFCFF00000000FAFF00000200FDFF0200000000000300FEFF0200FFFF03000000FEFFFDFF06000400FEFFFFFF0300FEFF00000400FDFF0100000001000200FEFFFAFF0100FDFF0100FFFFFBFF0200FDFFF9FF010002000500FFFFFFFFFDFFFFFF0000"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[3, 0, 1]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0x00000200FDFFFDFF010001000000FDFF0400FDFF00000100FBFF000005000000FFFF0000FEFFFAFFFFFF00000300FEFF0000FEFFFEFF0300FEFF0300FCFF0000FEFFFDFF05000200FEFF0000040000000000FBFFFFFFFFFF000002000000FDFF02000500FEFFFEFFFCFFFEFF03000300FEFF02000000FDFFFFFF04000100FCFFFFFFFEFFFBFF020000000100000007000000FCFFFDFF0000FBFFFDFF05000200FDFF00000000FFFF0100000000000000030002000100FEFF0000F7FF0100FDFF0300000001000100FEFFFCFF00000000FAFF00000200FDFF0200000000000300FEFF0200FFFF03000000FEFFFDFF06000400FEFFFFFF0300FEFF00000400FDFF0100000001000200FEFFFAFF0100FDFF0100FFFFFBFF0200FDFFF9FF010002000500FFFFFFFFFDFFFFFF0000"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

