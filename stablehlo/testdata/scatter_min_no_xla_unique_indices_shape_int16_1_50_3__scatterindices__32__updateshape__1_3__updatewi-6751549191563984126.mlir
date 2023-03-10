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
    %0 = stablehlo.constant dense<"0xFEFF01000400FEFF0000000002000100FDFF00000600000000000000020000000100FFFF03000300FAFFFFFF02000100FDFFFEFF040001000500FFFFFBFF00000200010001000300000000000100FFFF0100FBFF0200FEFF0000060002000400FEFF000003000000FDFF0000FFFF0500020000000000FAFFFCFF0200000003000000FEFF0000F8FFFCFFFDFFFFFF02000000FDFF000000000200FFFFFFFFFFFFF7FF03000200FDFFFDFF080001000100FCFFFFFFFFFFFFFF040003000000FEFFFAFF0200FEFF030001000200FEFF00000000FEFF00000000FEFFFFFFFCFF000004000300FEFFFAFF010001000000000000000100FDFF000004000200FFFF00000200FEFF0200F7FF0200010005000000FEFFFFFFFFFF000003000000FFFFFDFFFFFF0100010000000000FDFF"> : tensor<1x50x3xi16>
    %1 = stablehlo.constant dense<[[-3, 0, 2]]> : tensor<1x3xi16>
    return %0, %1 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> tensor<1x50x3xi16> {
    %0 = stablehlo.constant dense<"0xFEFF01000400FEFF0000000002000100FDFF00000600000000000000020000000100FFFF03000300FAFFFFFF02000100FDFFFEFF040001000500FFFFFBFF00000200010001000300000000000100FFFF0100FBFF0200FEFF0000060002000400FEFF000003000000FDFF0000FFFF0500020000000000FAFFFCFF0200000003000000FEFF0000F8FFFCFFFDFFFFFF02000000FDFF000000000200FFFFFFFFFFFFF7FF03000200FDFFFDFF080001000100FCFFFFFFFFFFFFFF040003000000FEFFFAFF0000FEFF030001000200FEFF00000000FEFF00000000FEFFFFFFFCFF000004000300FEFFFAFF010001000000000000000100FDFF000004000200FFFF00000200FEFF0200F7FF0200010005000000FEFFFFFFFFFF000003000000FFFFFDFFFFFF0100010000000000FDFF"> : tensor<1x50x3xi16>
    return %0 : tensor<1x50x3xi16>
  }
}

