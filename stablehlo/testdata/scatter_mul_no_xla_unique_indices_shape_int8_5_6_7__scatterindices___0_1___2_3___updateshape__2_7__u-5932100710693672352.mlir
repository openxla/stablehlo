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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2xi32>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<2x7xi8>) {
    %0 = stablehlo.constant dense<"0x0004FF000203FFFFFDFEFF00FE00030303FF01000002050100FEFF030000FCFBFFFD04FC0200FC0005FD000300000100FDFFFD0003050001FF0400000101FE06FF00FD00040100020501FEFDFEFD0502020102FD02FDFD00FFF8FE0000FDFEFFFD04000000FE05FBFE03040004FB01FF00FC000402020004FE01FC0100FDFD03FF0000FDFFFE02FEFE030303020002FAFD000103FF0005FD0100F80100FE03030001020003FF08FEFD0200FD03000005FE00FC02FE000100FE00FCFBFC0200FF02000005FFFF000201FBFEFDFFFC000201FF"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[-3, 0, 3, -3, 4, -5, 0], [-3, -3, 3, 0, 0, -1, -2]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x0004FF000203FF0300FA03000A00030303FF01000002050100FEFF030000FCFBFFFD04FC0200FC0005FD000300000100FDFFFD0003050001FF0400000101FE06FF00FD00040100020501FEFDFEFD0502020102FD02FDFD00FFF8FE0000FDFEFFFD04000000FE05FBFEF7F4000000FF0200FC000402020004FE01FC0100FDFD03FF0000FDFFFE02FEFE030303020002FAFD000103FF0005FD0100F80100FE03030001020003FF08FEFD0200FD03000005FE00FC02FE000100FE00FCFBFC0200FF02000005FFFF000201FBFEFDFFFC000201FF"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

