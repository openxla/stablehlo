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
      %5 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0x00FCFC0002FD00FC0000FD00FD030002FE00020000FFFF00FAFD02FDFB01FE00020000FBFFFEFD00010100FEFC01FCFF02FCFFFCFD050301FE000005FC040106FF00FF00FBFF0101FFFEFF00FF04000400020000080003000202FFFE0100FF000205FF04FE040000050203FCFC0200FE00FFFF0200FE03FB0101FEFD0001010002FC01FF01010000FF000202FDFD00FC0100FCFFFF00FEFB00FBFEFE010003FFFF000000FFFE00FCFE03FE0502FF03FE010000F70000FF0001FE030503FD03FE03FB00040000FC000000FD020000FEFF0601"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0x01FD00030301FF01020200FF00FD07030101FA00FD00070103FD00FDFFFB0203FEFE0000FF03060000FC0305FE0100FD03FB07FBFF02FC0007FB0000020401FCFC020104FF0000FE00030502FD00FF020102FF00FC00FF04FF000003FFFF010001FDFFFA000000FC03FEFD01FEFF07FE03FF000001FB000006FDFFFDFBF800FCFC01FF00FF00FE0002FF05FC"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x01F9FC0305FEFFFD0202FDFFFD000705FF01FC00FDFF0601FDFA02FAFB01FE00020000FBFFFEFD000101FFF9FE04FAFD02FCFEFF030503FD0105FE06FC01040106FBFE02F7FF0101FFFEFF00FF040004000200000FFB0300040600FAFD0200040105FF02FE070502020202FEFD04FFFE00FFFF0200FE03FB0101FEFD0001FD00010000FF0104FFFF000003FFFCF700FC01FCFFFDFC01FCFA07F9FEFE010003FFFF000000FFFE00FC0102FE0503FA03FE07FDFFF4FBF8FFFCFDFF020502FD01FE05FA05000000FC000000FD020000FEFF0601"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

