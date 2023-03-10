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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi16>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x125xi16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi16>, tensor<1x125xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi16>, tensor<1xi16>) {
    %0 = stablehlo.constant dense<"0xFAFFFEFFFEFFFFFFFEFF0100FFFFFEFF0100F9FF0300FFFF060000000300FBFF04000100F6FF02000700FEFFFFFF0200FFFF000003000300010000000500FDFFFFFF00000200FEFF010006000100FFFF0100FFFFFFFF02000200020001000100FDFFFFFF000000000000FDFF0000000006000300FEFF0000FFFFFFFF03000000FBFFFFFF06000200FDFF010004000100020005000000010002000100FEFF0000FEFF00000000FCFFFDFFFEFF0200FDFFFDFF0200FFFF04000000000000000500FBFF06000200F9FF000000000000FCFFFEFF04000300FFFFFDFF0000000000000000FBFF01000400F9FFFEFF0100000005000100FCFF0000FCFF"> : tensor<1x125xi16>
    %1 = stablehlo.constant dense<0> : tensor<1xi16>
    return %0, %1 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> tensor<1x125xi16> {
    %0 = stablehlo.constant dense<"0xFAFFFEFFFEFFFFFFFEFF0100FFFFFEFF0100F9FF0300FFFF060000000300FBFF04000100F6FF02000700FEFFFFFF0200FFFF000003000300010000000500FDFFFFFF00000200FEFF010006000100FFFF0100FFFFFFFF02000200020001000100FDFFFFFF000000000000FDFF0000000006000300FEFF0000FFFFFFFF03000000FBFFFFFF06000200FDFF010004000100020005000000010002000100FEFF0000FEFF00000000FCFFFDFFFEFF0200FDFFFDFF0200FFFF04000000000000000500FBFF06000200F9FF000000000000FCFFFEFF04000300FFFFFDFF0000000000000000FBFF01000400F9FFFEFF0100000005000100FCFF0000FCFF"> : tensor<1x125xi16>
    return %0 : tensor<1x125xi16>
  }
}

