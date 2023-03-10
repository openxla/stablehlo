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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi16>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x125xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi16>, tensor<1x125xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi16>, tensor<1xi16>) {
    %0 = stablehlo.constant dense<"0xFBFF0000FFFF00000000FEFFFCFFFAFF02000200010001000000000000000000FFFF000008000200000005000000030000000000FCFF0300FEFFFFFF00000000FEFF0000FDFFFFFF0100FFFF0100FCFF03000000FFFF0100FDFF04000100FEFFFEFFFEFF0000FCFFFEFF00000000050002000000020003000000FCFF0000FBFF0200FDFF03000300060000000200FDFF00000100FFFF01000300010001000000050001000200FEFFFFFF01000100FFFF04000300FFFFFFFFFAFF02000200030000000100FFFF0100FDFFFDFF000001000000000000000000000000000000FFFF00000000060000000200FEFF0400FDFFFFFF0000040003000200"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<-2> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0x0A000000FFFF00000000FEFFFCFFFAFF02000200010001000000000000000000FFFF000008000200000005000000030000000000FCFF0300FEFFFFFF00000000FEFF0000FDFFFFFF0100FFFF0100FCFF03000000FFFF0100FDFF04000100FEFFFEFFFEFF0000FCFFFEFF00000000050002000000020003000000FCFF0000FBFF0200FDFF03000300060000000200FDFF00000100FFFF01000300010001000000050001000200FEFFFFFF01000100FFFF04000300FFFFFFFFFAFF02000200030000000100FFFF0100FDFFFDFF000001000000000000000000000000000000FFFF00000000060000000200FEFF0400FDFFFFFF0000040003000200"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

