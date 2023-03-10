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
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2xi32>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<2x7xi8>) {
    %0 = stablehlo.constant dense<"0x04000300FEFFFF01FF000700FEFF010001040504FD0003FC0003FE00000408FDFF0001FF01FF03FE0001FB0303FD0201FEFF0003FCFEFAFE000403FBFD000000FC00FB0002040001FC0100FF0000FDFDFE02FDFE04FD020000FE01010004FFFC0200FC04FAFE0100FF000404000702000006FF00020100030100000500FDFAFEFD010002FD0100FF000000FFFD00FE0000FDFFFFFEFE02FE000000FF000100FF0202FC0501FE000000FF01FDFF0000000305FFFD0003FE0000FFFFFF0000FF00FF00FF00000001FEFF0302FDFEFDFDFE0401"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[1, 0, -1, -1, 0, 1, 3], [-1, -4, 2, 0, 0, 0, -4]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x04000300FEFFFF0100FFFF000103010001040504FD0003FC0003FE00000408FDFF0001FF01FF03FE0001FB0303FD0201FEFF0003FCFEFAFE000403FBFD000000FC00FB0002040001FC0100FF0000FDFDFE02FDFE04FD020000FE01010004FFFC0200FC04FAFE0100FFFFFC02000000FC0006FF00020100030100000500FDFAFEFD010002FD0100FF000000FFFD00FE0000FDFFFFFEFE02FE000000FF000100FF0202FC0501FE000000FF01FDFF0000000305FFFD0003FE0000FFFFFF0000FF00FF00FF00000001FEFF0302FDFEFDFDFE0401"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

