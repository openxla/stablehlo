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
      stablehlo.return %arg1 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2x1xi64>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFC01030004FD0001FA0102FEFF00FB00FDFA00FF00FD0100000000FF00FE0100020400000000FC0200FEFFFC040101FBFE0000FEFEFFFC0100FF000400FD00FCFD00FA000300FAFE01FD0001FF0700FEFD030302FD01FEFE00FF020401FE0401FC0004FE01000004FE0100FCFEFFFC02000701FD0300FEFB03FF0000FE05040000FF0502FDFFFE01FF0300FD00010101FFFD0003FEFAF903FDFC0001FF0100FE0100FFFDFFFB0000FF000001FE0100000400000002FD04FEFBFEFEFA0100FF030100FCFD0201FA00030500FE02FD0100F903"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<"0x00FE08000500FF0103F8FEFC020000050000FD000001FDFC01010001FF00FDFE00040001FFFB00F903FF00050000FE000000FDFE02FEFE0000000000060500FFFCFFFEFC00FB00020006FCFE0000FE00FF00010000FDFE000001FF00FFFE00FCFD0301FE04000000020200FC0200FD0000FC0201FFFE0302050101010008000003FC0002000200FE0303FD00"> : tensor<5x2x2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00FE08000500FF0103F8FEFC020000050000FD000001FDFC0101000100FE0100020400000000FC0200FEFF00FDFE00040001FFFB00F903FF00050000FE000000FDFE02FEFE00FAFE01FD0001FF0700FEFD03030200000000060500FFFCFFFEFC00FB00020006FCFE0000FE00FF000100000701FD0300FEFB03FF0000FE0500FDFE000001FF00FFFE00FCFD0301FE04000000020200FC0200FD000001FF0100FE0100FFFDFFFB000000FC0201FFFE0302050101010008000003FC0002000200FE0303FD000201FA00030500FE02FD0100F903"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
