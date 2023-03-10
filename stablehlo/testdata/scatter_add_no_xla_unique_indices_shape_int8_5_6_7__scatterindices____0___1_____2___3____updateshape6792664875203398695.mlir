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
    %0 = stablehlo.constant dense<"0x02020001FE02FF00FE0002FF000001020001040003FF0100FF0004FAFCFC04FD00FDF5FFFE000006020200000400000200FDFEFEFEFC00050105FFFC010405FB00FA0003FFF803FF0002FFFF00FB0300000301020005040104FD02000100FE02FF02030503FBFFFC00FE00FCFF00050104FDFD01FFFFFE03010402FDFE000101FF00FEFF0302040001FB0300F903FFFE0004FDFC0001FC020000020106010003FB0004FF0002FF02FE0101FE03FD010301FFF80600F801FF00FD06FFFD04FF00FA0002FD06F60202F7FDFF0006FF020000FF"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0xFFFEFFFB0000FF02FF0000FEFFFC0000FF00FC00FEFD0003010004000501010101FE010403010401000800050301FD06000002FCFD00FDFD04FF000202FD00020000000300FE0204FFFFFFFFFC00010003000302FF0003FE0000FEFE030402FF03FE00020003FF0001010204FFFDFFFE01FD00000301FFFE0003FE0200FF03FEFE05FE03FE0004FE01FFFC00"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x0100FFFCFE02FE02FD0002FDFFFC0102FF01000001FC0103000008FAFCFC04FD00FDF5FFFE0000060202050105010100010101FF02FD000D010A02FDFE0A05FB02F6FD03FCF503FF0002FFFF00FB0300000301020404040306FA02020100FE05FF00050902FAFEFBFCFE01FC0200080304FDFD01FFFFFE03010402FDFE00000102FEFEFF0100070403FA06FEF905FF01FF04FEFD0205FBFFFFFE020106010003FB0004FF0002FF02FFFE01FE06FE00010102F60800F704FDFE020402FB0403FEFBFFFEFD06F60202F7FDFF0006FF020000FF"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

