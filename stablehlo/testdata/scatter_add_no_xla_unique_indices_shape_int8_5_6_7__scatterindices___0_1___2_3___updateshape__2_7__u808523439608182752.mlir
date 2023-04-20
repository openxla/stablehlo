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
    %0 = stablehlo.constant dense<"0x02000003000200FC02FCFD0200000100FEFB01F9FD03FEFDFF03FC0000020000FE0200000101FD000101000600FDFEF8FFFDFF01030000000101FD0000000204FAFF0000FA0000000104020506FF0402FF03FF00FEFEFC0200010700FCFFFD0001FDFB0000FA02020105FD000000FFFD0203FD01000502FB0104FF01FDFFFE02FF00FFFEFF00010007020102FFFE0102FF00000004000400040300030100FE01FF000002FEFDFF0400000000FD00FE02020000000000FE0000FF00FFFF00FF00FEFEF8000205FE02FE0602FE0603000307FE"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[2, 0, -3, 1, 0, 0, 0], [4, -1, 0, -4, 2, -1, -8]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x02000003000200FE02F9FE0200000100FEFB01F9FD03FEFDFF03FC0000020000FE0200000101FD000101000600FDFEF8FFFDFF01030000000101FD0000000204FAFF0000FA0000000104020506FF0402FF03FF00FEFEFC0200010700FCFFFD0001FDFB0000FA02020109FC00FC02FEF50203FD01000502FB0104FF01FDFFFE02FF00FFFEFF00010007020102FFFE0102FF00000004000400040300030100FE01FF000002FEFDFF0400000000FD00FE02020000000000FE0000FF00FFFF00FF00FEFEF8000205FE02FE0602FE0603000307FE"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

