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
    %0 = stablehlo.constant dense<"0x00FCFF00FDFC0002FE03FC00000004FEFFFF0001000301FD01FE0100FF010102050000FF0302FCFE02FEFDFF00FE00FF0201FBFBFE01FF00FF010000FF0100FE080100030400F901FD0402040000FF02FF0300FFF9FD0107FD03FE020000050202FF01FE000802000202FD02FDFE0002FFFF000102FF0000FF00FC030000FE01FCFDFC00000101FF0300FF02FEFD00FAFF0206010300FC01FDFF0005060001000400FB0104FD00000503F90401000102FDFF00FD01060100FD0203050004FC000001010000FFFF00FA0102000001000000F9"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[5, 2], [2, 0]], [[0, -1], [-1, 2]], [[0, -2], [-1, 5]], [[2, -1], [5, -9]], [[2, -2], [3, 1]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x0005FF00FDFC0002FE00FC00000004FEFF020001000301FD01FE010002010102050000FF0302FCFE02FEFD0000FE00FF0201FB02FE01FF00FF0100FFFF0100FE080100030400FF01FD0402040000FF02FF0300FFF9000107FD03FE020005050202FF01FE00FE02000202FD02FDFE0002FFFF000102FF0000FF00FC030000FE02FCFDFC00000101F70300FF02FEFD00FFFF0206010300FC01FDFF0505060001000400FB0104FD00000502F90401000102FD0100FD01060100FDFE03050004FC000001010003FFFF00FA0102000001000000F9"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

