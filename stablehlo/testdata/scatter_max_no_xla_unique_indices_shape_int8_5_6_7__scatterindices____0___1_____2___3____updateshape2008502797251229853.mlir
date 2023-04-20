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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0x01FD06FF070100FEFF0404FC0000FAFE00000503030006FE00FE06FF010000FFFBFF0500FDFEFE0406070000FDFF03FEFEFE000000FFFE00FFFEFC04FF03030200FD01FD0101030200FF05FC00FE02FC00FBFEFC00FDFF010001000202FEFF01050200FFFF0200000202FFFD0000FF04FF01FF04040200FDFFFF0200FF000107FF0501020008FE010001FEFEFF0401FEFEFCFC0303FDFFFC00FC07FF0301030000FE03FFFFFCFF01010305FF0206FC010200020104FC0703FC000000000206FFFFFF000001FC0104FA00FEFC0400000003FF"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0x0001FCFA02070000FE02FE010200FE01FFFCFE00FE0103FF0402010000FD01FE00000001FE000100FDFEFC00FE0002FDFF0300000000010604FFFE05FF01FEFE01FF05010006FD02FD0200FF02FE00FE010803FCFC03010003020101000000FE00FCFE02FDFB0304FD0100FDFDFF04FEFAFEFD02020102FE0203FEFB0200000200010000FDFF00FFFD0502FD"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x010106FF07070000FF0404010200FE0100000503030106FF04020600010000FFFBFF0500FDFEFE040607000001FF0300000100000100FE00FF00FE0402030303000001000106030200FF05FC00FE02FC00FBFEFC04FFFF050001000202FF050105060002FF020000020200FE01080304FF01FF04040200FDFFFF0200FF0001070105030201080001000100FEFF0401FE0304FD0303FDFFFF04FE07FF0301030000FE03FFFFFCFF010103050202060201020302010400070300010000000206FFFF05020001FC0104FA00FEFC0400000003FF"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

