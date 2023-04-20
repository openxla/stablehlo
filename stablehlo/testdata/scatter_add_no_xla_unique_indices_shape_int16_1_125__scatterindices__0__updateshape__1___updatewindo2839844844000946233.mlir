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
    %0 = stablehlo.constant dense<"0xFFFF0000000000000000F7FF0100030000000000FEFFFCFF0000020002000000FEFF00000200FBFFFEFF00000200FCFF0300FEFF0000FDFFFEFF0000FFFF0100050009000100FEFF0000030006000A000100FCFF0000FAFF0000FFFFFDFFFCFFFEFF010002000200FFFF0000040000000100FCFF00000000FFFF0000FCFFFFFF0400FDFFFDFF0300F9FFFDFF020000000500FEFFFFFF0000FFFF00000000FFFFFDFFFEFFFFFF0300FDFF000001000000FDFF02000200000001000100FDFF000001000300FFFF04000700020001000000FCFFFEFFFCFF00000600FDFF0100FAFFFBFFFFFF02000600FDFF060000000300FFFFFDFF0000FBFF0000"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<-5> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0xFAFF0000000000000000F7FF0100030000000000FEFFFCFF0000020002000000FEFF00000200FBFFFEFF00000200FCFF0300FEFF0000FDFFFEFF0000FFFF0100050009000100FEFF0000030006000A000100FCFF0000FAFF0000FFFFFDFFFCFFFEFF010002000200FFFF0000040000000100FCFF00000000FFFF0000FCFFFFFF0400FDFFFDFF0300F9FFFDFF020000000500FEFFFFFF0000FFFF00000000FFFFFDFFFEFFFFFF0300FDFF000001000000FDFF02000200000001000100FDFF000001000300FFFF04000700020001000000FCFFFEFFFCFF00000600FDFF0100FAFFFBFFFFFF02000600FDFF060000000300FFFFFDFF0000FBFF0000"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

