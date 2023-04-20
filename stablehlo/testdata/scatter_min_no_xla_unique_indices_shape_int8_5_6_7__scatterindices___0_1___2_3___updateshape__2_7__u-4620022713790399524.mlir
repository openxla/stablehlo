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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2xi32>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<2x7xi8>) {
    %0 = stablehlo.constant dense<"0x00FD01FEFFFDFA0000FE000001FFFE01FD00020400FB0102020202FBFE0303FD00F9FDFCFFFFFCFDFD00FE05FF0A0103FB0500FE00FAFE03FDFF06FF0000000408FEFF03FD03FFFE00FE0000FFFF0104000201090000FB000202FE04FEFF03FBFD05FEFE00FDFF03FFFFFEFF0101000002FE03030000000402030007FE0000FFFFFFFFFAFD0001030100000201FB00010000FE00040004000000FD01FFFE0106030103FD00FF02FF01000301FF0001FA00FE00FFFC00FD050202070000FBFF02FDFE00FF04FD0100000101FFF90400FB0104"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[-4, -1, 4, -1, 1, 1, 0], [1, 1, -1, 4, -1, 2, 3]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x00FD01FEFFFDFAFCFFFEFF0001FFFE01FD00020400FB0102020202FBFE0303FD00F9FDFCFFFFFCFDFD00FE05FF0A0103FB0500FE00FAFE03FDFF06FF0000000408FEFF03FD03FFFE00FE0000FFFF0104000201090000FB000202FE04FEFF03FBFD05FEFE00FDFF03FFFFFEFF01FF000002FE03030000000402030007FE0000FFFFFFFFFAFD0001030100000201FB00010000FE00040004000000FD01FFFE0106030103FD00FF02FF01000301FF0001FA00FE00FFFC00FD050202070000FBFF02FDFE00FF04FD0100000101FFF90400FB0104"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

