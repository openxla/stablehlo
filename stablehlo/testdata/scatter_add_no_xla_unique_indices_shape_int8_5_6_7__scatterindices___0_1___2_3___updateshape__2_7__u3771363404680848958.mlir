// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2xi32>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<2x7xi8>) {
    %0 = stablehlo.constant dense<"0xFF0200FD03FC00FF00FC01FFFAFFFC0001020400010001010000FE0400FF00050000070001000000FBFB0001FD000400060000FD0302FA010100FAFC0003040200F804FE00FFFCFA04FFFF00F9FF0200FB0001010501FE0000FD0000FC0002FE01FD00F905F9FF03FEFFFFFA01000100F9FF00FFFDFF04FF00FFFFF9000001FEFEFB01FF0000FEFC00FE04FF01FD03010104FD020300FE0201FE01FD00FF00FFFE00FFFD0001FBFF0103FB05FFFEFCFFFEFD00FC0402000000FF0000FEFD0102FEFE010100FDFAFBFE0001FF00FFFE000004"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[-2, 0, -3, 3, 2, -2, -8], [-4, 0, -2, -1, 3, -2, 0]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFF0200FD03FC00FD00F90401F8F7FC0001020400010001010000FE0400FF00050000070001000000FBFB0001FD000400060000FD0302FA010100FAFC0003040200F804FE00FFFCFA04FFFF00F9FF0200FB0001010501FE0000FD0000FC0002FE01FD00F905F9FF03FEFBFFF80003FF00F9FF00FFFDFF04FF00FFFFF9000001FEFEFB01FF0000FEFC00FE04FF01FD03010104FD020300FE0201FE01FD00FF00FFFE00FFFD0001FBFF0103FB05FFFEFCFFFEFD00FC0402000000FF0000FEFD0102FEFE010100FDFAFBFE0001FF00FFFE000004"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

