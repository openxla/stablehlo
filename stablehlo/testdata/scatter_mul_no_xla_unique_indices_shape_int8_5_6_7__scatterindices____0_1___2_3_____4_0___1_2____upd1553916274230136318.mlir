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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0xFEFDFC040003FDFE0200FBFF0002010003FE000003010100050500FD01000101FDFF02FEFE00FEF9030400FF06000001FFFE010501020501FBFF0103FD0204FF01FB00FD020002020BFF000107FCFE01FFFBFF01FF0000020002FE0403FCFE0003FB03FA00FEFC00040505010101000104FE00FD0100FE00FF0200FCFC000301FF04FC0101FD0000030201010101FE05FD0203FEFBFF00000305010100FFFF04FBFE01000502FDFE03FF00FD0000FE040001FD00FD03000000FE000400050000010005FF0500FFFE0001FDFD000000000104"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[0, 1], [3, -1]], [[-1, 3], [-1, 4]], [[-2, 1], [-8, -2]], [[1, -3], [2, -1]], [[0, -1], [0, 0]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFE00FC040003FDFE0200FBFF0002010003FE000003010100050500FD03000101FDFF02FEFE00FEF90304000106000001FFFE011401020501FBFF0109FD0204FF01FB00FD0200FE020BFF000107FCFE01FFFBFF01FF0000020002FE040308FE0003FB03FA00FEFC000405050101010001E0FE00FD0100FE00FF0200FCFC000301FF04FC0101FD0000030201010101FEF1FD0203FEFBFF00000305020100FFFF04FBFE01000502FDFE030000FD0000FE040000FD00FD0300000002000400050000010005FF0000FFFE0001FDFD000000000104"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

