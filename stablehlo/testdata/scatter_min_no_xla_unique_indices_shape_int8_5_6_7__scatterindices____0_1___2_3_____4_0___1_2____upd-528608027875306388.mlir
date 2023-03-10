// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0xFEFE05FD02FF000002FC03FE0204FFFC00FE0000000AFD0000FE0100000004FEFE00FEFE00FD00FB040001FFFEFE00000001FD020001FF03000104FCF70000FB020100FEFF00FEFDFE000000F901020100010707FE02FD04FF05FF0401FE030000FE01FC00FF03FDFEFFFEFAFFF9FF00FFFF02000000000200FBFE0205FFFE00020102FEFD07FE000000FDFB06FF0301FE0000FC030500FFFE00FC04000001FEFBFDFEFD020000020006FF0302FD00FC020204FA00FDFFFEFDFD000103FAFE01FE040300FFFEFD03000203020004FF000004"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[0, -1], [3, 6]], [[5, 0], [7, 8]], [[0, -2], [3, 3]], [[-5, -2], [-2, -3]], [[0, -1], [5, 0]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFEFE05FD02FF000002FC03FE0204FFFC00FE0000000AFD0000FE0100000004FEFE00FEFE00FD00FB040001FFFEFE00000001FD020001FF03000104FCF70000FB020100FEFF00FEFDFE000000F901020100010707FE00FD04FF05FF0401FE030000FE01FC00FE03FDFEFFFEFAFFF9FF00FFFF02000000000200FBFE0205FFFEFB020102FEFD07FEFD0000FDFB06FF03FEFE0000FC030500FFFE00FC04000001FEFBFDFEFD020000020000FF0302FD00FC020004FA00FDFFFEFDFD000103FAFE01FE040300FFFEFD03000203020004FF000004"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

