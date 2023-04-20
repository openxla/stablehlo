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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi16>, tensor<1xi32>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16>, tensor<1x3xi16>) {
    %0 = stablehlo.constant dense<"0x000008000200FFFF06000200FFFF0300FDFF00000000FFFF0200FFFFFBFF00000000020000000200040004000000FFFF000000000000FAFFFCFF010002000100FFFF0300FFFF00000200FBFF00000000FEFF03000300FDFF0100000003000000FCFF00000000000000000000FDFF0000FFFF00000100FFFF02000000FFFF0000000000000100FDFF0000FCFF0600FEFF00000300FDFF05000300000002000100FDFF0200FEFF030000000200030001000200FDFFFFFFFFFFFCFF02000400FFFF030000000000F9FF0100000000000000FFFF02000000FDFF04000000030000000000FEFFFEFF060000000100FEFF010007000000FFFFFDFF000004000400FDFFFDFF01000100FCFF0200FFFF0000FDFFFFFF000002000100FEFF0000FDFF0000F8FF0200FDFF0300FFFF0000"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[3, -5, 1]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0x000008000200FFFF06000200FFFF0300FDFF00000000FFFF0200FFFFFBFF00000000020000000200040004000000FFFF000000000000FAFFFCFF010002000100FFFF0300FFFF00000200FBFF00000000FEFF03000300FDFF0100000003000000FCFF00000000000000000000FDFF0000FFFF00000100FFFF02000000FFFF0000000000000100FDFF0000FCFF0600FEFF00000300FDFF05000300000002000100FDFF0200FEFF030000000200030001000200FDFFFFFFFFFFFCFF02000400FFFF0300FBFF0000F9FF0100000000000000FFFF02000000FDFF04000000030000000000FEFFFEFF060000000100FEFF010007000000FFFFFDFF000004000400FDFFFDFF01000100FCFF0200FFFF0000FDFFFFFF000002000100FEFF0000FDFF0000F8FF0200FDFF0300FFFF0000"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

