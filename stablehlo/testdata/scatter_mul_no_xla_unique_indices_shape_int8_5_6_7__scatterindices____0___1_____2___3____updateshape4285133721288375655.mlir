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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0x00FD00000500FB0200FF04FFFD020000FEFCFF00FFFEFCFF01000102FF0000FD0202020700000101FFFDFFFD000001000303FE01000100FF0000FFFD010200FE01FFFF01FFFF0300FF0404FDFE00FC07FEFEFD030303000504FF00000002FF04FD00FFFC01FF0200010000FD00FF01FF060104FF0002FD04FCFC030000FE0000020001FF02FD04FF00FFFC04FE00FD010000FFFEFD02FF000003040303FE000402050203FEFE0300FC000000FC000001050402000400FF000001FDFEFF000003000106000000FBFE00FF0000FFFF03FF01FF"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0xFB01010000FD01FCFFFB0101FF01FFFF0001FFFCFC02FE00010100FE01FD0000020500FEFF06FD02FF00020400FDF9FD01FDFF00FE0201FD030200020100FF0002FDFF000002FFFB0000FEFF01FD0101FFFD0100FCFF010001FC0200FF0000FFFFFFFFFF0300000002FC000102060402FCFF00FE030502FF020000FEFD03F5000006040203FF00010100FD02"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x00FD00000000FBF8000504FF0302000000FC010004FC0800010000FCFF0000FD0202020700000101FFFDFF090000020000FA02060002000000000009F9FA0006FF000202FF030300FF0404FDFE00FC07FEFEFD030906000A0400000000FA0100000001140000FC00010000FD00030100060104FF0002FD04FCFC030000FE0000020001040400FC00000104FC0200F7000000FE080002FE000006040303FE000402050203FEFE030010000000F40000FF0A000000F4000B000006F4FCFD0000030000EE000000FBFE00FF0000FFFF03FF01FF"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

