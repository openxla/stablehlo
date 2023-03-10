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
      stablehlo.return %arg1 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi16>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x125xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi16>, tensor<1x125xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi16>, tensor<1xi16>) {
    %0 = stablehlo.constant dense<"0x0400010001000000FDFF00000300FEFF0400FCFFFEFF00000000FEFF020001000100000002000200FEFF0000FFFFFFFF00000300FFFF0000FEFFFBFF00000300FDFFFEFFFEFFFDFF050003000000FFFFFDFF00000200020000000100FDFFF9FF0000FFFF0100FEFF05000600FBFF040000000100FDFF0400020001000200FEFF010006000600050001000000FFFF03000000FEFFFDFFFFFFFFFF02000000FBFF04000000FFFF0300FDFF000001000200FFFF04000100FCFFFDFF0000000007000200FDFF0000FEFF00000000FAFF00000100FFFF0000FFFFFCFF00000300010004000000020000000400FEFF0300FCFF01000600030002000600"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<-1> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0xFFFF010001000000FDFF00000300FEFF0400FCFFFEFF00000000FEFF020001000100000002000200FEFF0000FFFFFFFF00000300FFFF0000FEFFFBFF00000300FDFFFEFFFEFFFDFF050003000000FFFFFDFF00000200020000000100FDFFF9FF0000FFFF0100FEFF05000600FBFF040000000100FDFF0400020001000200FEFF010006000600050001000000FFFF03000000FEFFFDFFFFFFFFFF02000000FBFF04000000FFFF0300FDFF000001000200FFFF04000100FCFFFDFF0000000007000200FDFF0000FEFF00000000FAFF00000100FFFF0000FFFFFCFF00000300010004000000020000000400FEFF0300FCFF01000600030002000600"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

