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
    %0 = stablehlo.constant dense<"0xFAFF0400030003000000020002000200FFFF0000FDFFFCFF0200FDFF00000200FCFF0000FDFFFEFFFDFF0400FDFF000005000000FFFF020000000100FDFF030007000000000000000100FEFFFFFF030000000000FFFF00000300FCFFFEFF000001000000FFFFFBFF0000FDFF030000000000FBFF01000000000001000000FDFF00000000060003000800FEFF050000000400FFFFFAFF02000000050000000600FDFFFAFFFFFF0000030003000000FDFFFFFFFEFF0000FEFF0000FCFFFEFF0000FDFF0000FDFF0000FCFF02000200FBFF0500000000000400FBFF020000000000FDFF02000000FFFF0000FFFF0200000000000200030004000100"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<-1> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0x06000400030003000000020002000200FFFF0000FDFFFCFF0200FDFF00000200FCFF0000FDFFFEFFFDFF0400FDFF000005000000FFFF020000000100FDFF030007000000000000000100FEFFFFFF030000000000FFFF00000300FCFFFEFF000001000000FFFFFBFF0000FDFF030000000000FBFF01000000000001000000FDFF00000000060003000800FEFF050000000400FFFFFAFF02000000050000000600FDFFFAFFFFFF0000030003000000FDFFFFFFFEFF0000FEFF0000FCFFFEFF0000FDFF0000FDFF0000FCFF02000200FBFF0500000000000400FBFF020000000000FDFF02000000FFFF0000FFFF0200000000000200030004000100"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

