// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0x00FF03FE0304FBFCFCFD000503FD0202FFFDFF00FFFFFFFD01FD00030403FE02050003020101FDF9FCFEFFFBFFFE030102FC05020300FF0201FE010101FE000203FFFFFF000000FDFA0200000402FFFD03010102FC05FE00FBFDFD02FC0200FBF8FAFAFE000305FFFFFD0203000401FF040200FDFC01FFFF0005FB08FE03FE00040000FEFD000300FD000301FC00FC040002FF0001FFFFFEFF03FC0501FA04040007FDFF02FF030102FEFF02FF04FD0001FF01FF00FE00FE0203FE000200FF05FF00FFFF0101FFFBFD0101FF00FCFE000001"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0x00FF0100FF0101000400FF00020301020000FAFE04FF00FAFF04FE000102030100050200010001FC00FEFF0300FF01FD06F9FFFF04FC000000070000FDFFFF010300000204FC0000020202FF010000000100FFFD00020402FF01FC0000FE0005FF00FF00080303010003FEFF00FFFFFF02FA0203FDFDFDFDFE03FFFDFC000100010100FD000101FAFE0303FE"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x00FF03000304010004000005030302020000FF0004FF00FD010400030403FE02050003020101FDF9FCFE01020301030502000502030000020103010101FE060203FF04FF000000FDFA0200000402FFFD0301010200070000FDFFFF020302000204FC0000020305FF01000203010401FF040200FDFC01FFFF0005FB08FE03000204020001FD00030000050301FF0008040302000301FF00FFFF03FC0501FA04040007FDFF02FF030102FE0203FF04FD00010301FF000001000203000002010105FF0303FF0101FFFBFD0101FF00FCFE000001"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

