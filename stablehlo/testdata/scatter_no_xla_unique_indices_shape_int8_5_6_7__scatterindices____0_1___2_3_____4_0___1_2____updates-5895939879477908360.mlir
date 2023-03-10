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
      stablehlo.return %arg1 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0xFC030706000500FD020400030101FA000200FFFD0000040100FF0100FEFEFE00FDFE0102FC000602030400000203FC04FC03FDFBFFFEFE0000020002FEFEFF03FAFFFF00FEF8FE01FFFAFE00FEFFFD010001FFFB080100FAFFFE00FE02FC0500FE0600FF0002000003000100FF010200FFFD00FC00FA030002FEFD0503FFFF0200FDFFFD03FEFEFDFD03FC00030401FEFFF7FE000004FE01FB060102FE01FDFE0106FE03FA030002FEFCFDFFFA0400FCFFFEFE00FF020003FB01FF0202FCFE0000010302FF020102FE01FDFAFF0200FF0000"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[0, -3], [1, 1]], [[2, 1], [0, -6]], [[-1, 0], [-2, 0]], [[-9, 2], [-2, 0]], [[4, 2], [2, -4]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFC000706000500FD020100030101FA0002FDFFFD0000040100FF010001FEFE00FDFE0102FC000602030400020203FC04FC03FDFAFFFEFE0000020001FEFEFF03FAFFFF00FEF80001FFFAFE00FEFFFD010001FFFB08FF00FAFFFE00FE02000500FE0600FF0000000003000100FF010200FEFD00FC00FA030002FEFD0503FFFFF700FDFFFD03FEFE00FD03FC0003040102FFF7FE000004FE01FB06FE02FE01FDFE0106FE03FA030002FE04FDFFFA0400FCFFFCFE00FF020003FB02FF0202FCFE000001030202020102FE01FDFAFF0200FF0000"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

