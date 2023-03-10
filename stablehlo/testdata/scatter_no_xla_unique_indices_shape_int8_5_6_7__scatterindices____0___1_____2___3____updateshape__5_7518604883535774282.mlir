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
    %0 = stablehlo.constant dense<"0x0203FB06FC0000FFFFFD000600FD03FFFD01FFFE00FD01FF0203FF01FF00050200FDFF00010300040006FC030200010200000301000503FE02030000FFFA00FEFFFE0106000400FFFC00FD0100FD0300FD010000FE0305000002FB0000010303000403FF02FBF9FDFA01FF0100FBFD0205000200FC0302FF000401FE06FC01FE00FC00FE000004FFFF0300030601FDFF0201FEFFFF030101FE0001FD01FF0101FF02000101020003FF010202FDFE03030000FFFE05FE00FC00FF01FF00FD0000FFFBFD060400000104FFFE01030200FE0100"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0x040303FF0100FFFF000102FEFF00000401FF00010000FF04FFFB0200FF01000103FB0303000002FC030400FFFFFF0000FD00FF0100FF0400FE01FFFF00FE000001FFFDFE00FB0400FFFF0001010102FE0101FF01FE010000FD0200FFFEFAFF040001010101050000FCFEFE00000405010003FFFD00020000FE00F902FCFC02FE0302010000FEFD0000FD01FC"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x040303FF0100FFFF000102FEFF00000401FF00010000FF04FFFB0200FF00050200FDFF00010300040006FF01000103FB0303000002FC030400FFFFFF0000FD00FF0100FF040000FFFC00FD0100FD0300FD010000FE01FFFF00FE000001FFFDFE00FB0400FFFF0001010102FE0101FF0105000200FC0302FF000401FE06FCFE010000FD0200FFFEFAFF040001010101050000FCFEFE000004050101FD01FF0101FF020001010200030003FFFD00020000FE00F902FCFC02FE0302010000FEFD0000FD01FC0400000104FFFE01030200FE0100"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

