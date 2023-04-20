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
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0xFFFE0300FD000001FE000200FE02FEFC04FFFF000300FB06FD0300FE0000000200FFFDFDFE0908FD02FF00000200FC0100FF030005FE040502FFFC03FFFC00FB000404FC00FF05FC0103FD000302000000000301FF0103FF02FD00FF000100FF000000FF02FFFF01010101FD00020405000101FEFE000100FC0201FE02FB020100FAFFFD01FB000004000200070000FFFF0006010402FBFE020401F8FE00FE01FE00000402FE01010302010001FE01000001FC0001FEFAFD010300FD0103FF0100F705020003FE00FDFDFE0001FF0007FD02"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0x0000030205060000000601FFFE04010100FE0307FDFF0200FD000503FEFFFA00FB08030003FF00FEFDFB0100FB00000104FD03FD00FE03FEFAFF00FF02040103FE0202FB0002010200FEFBFEFEFF050101FE00FF0403FE03FD01FE01FEFE01FF010100FEFD0101020000FEFEFE02000003FB02FFFD000001000002FDFEFDFFFF05FC0102FEFF0204FE03FBFC"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x0000030205060000000601FFFE04010100FE0307FDFF0200FD0005030000000200FFFDFDFE0908FD02FFFEFFFA00FB08030003FF00FEFDFB0100FB00000104FD03FD00FE03FE05FC0103FD000302000000000301FAFF00FF02040103FE0202FB0002010200FEFBFEFEFF050101FE00FF000101FEFE000100FC0201FE02FB0403FE03FD01FE01FEFE01FF010100FEFD0101020000FEFEFE02000001F8FE00FE01FE00000402FE010103FB02FFFD000001000002FDFEFDFFFF05FC0102FEFF0204FE03FBFC0003FE00FDFDFE0001FF0007FD02"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

