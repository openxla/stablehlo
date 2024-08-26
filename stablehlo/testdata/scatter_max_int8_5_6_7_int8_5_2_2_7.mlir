// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>)
    %1 = call @expected() : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2x1xi64>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02FD01FFFFFF04FF03FFFF000403FEFEFD0001FFFE02FD00FCFB04FF010200FD0000FDFFFE01FE010502FE01FD00FF01FE04000007020100000100030005FE000301010300000000FF0002FD00FE0105FB000300FC0000040400000004FFFB0000FE0001010403FD00FE0205FFFDFC00020000FC000003FEFC00FE01010100FD0004FF02FD0100FFFA0200000403FCFB00050204FC01040200FEFF0202FEFDFF03FB0300FD00FEFEFC0001FEFC01000000FE00FD05FDFD0000FC00000002FDFFFD02FE04FE0506FF0100FFFCFEFE01000002"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<"0xFEFE02FB000001FC02FEFCFB0201FF02FDFF03FE0102030101FCFD0001000103FF00FBFDFE000103FD02FE00FFFD020300FE010306000103000004000000050002FDFE01FF06FFFE0000FDFD000004000600FFFFFD00FDFE05FE030200FE0201010101FCFC01FBFE010001FDFB0004FDFEFF01FE010102FEFD0400020100FBFB01FFFF00FD010005FEFCFDFF"> : tensor<5x2x2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02FE02FF000004FF03FFFF000403FF02FD0003FF0102030101FC0400010200FD0000FDFFFE01FE01050201010103FF01FE0400000703010200010003020500000303060301030000FF0002FD00FE0105FB000300000004040400050004FFFE0100060001010403FD000004050600FF00020000FC000003FEFC00FE010101000000040502030200FF020201010403FC01000502040101040204FEFF0202FEFDFF03FB0300FD00FEFEFE0001FE01010200000400020500FD0001FF000000020005FE02FE04FE0506FF0100FFFCFEFE01000002"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
