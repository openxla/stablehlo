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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0xFFFF07FF00FF030103020100FF03FE020300FD01FE02FC05FD000000000400FF02FCFAFF020000FA00FDFC040502FF02010000FFFD00050306FF00FF0400FFFCFCFE000305FD00FFFBFD05000202FE00010601010201FF02000103FFFDFEFD0102FA01FD01FC01FFFFFC050001FE00010000FE00000200FD00000001FFFE01FE0300FE0102FF000101FC03FE0101000703FD00FF02FEFE0101FD0002FF02FF010200FA01FEFF00FFFEFC000000FB04020402FEFA01F7FF02FF010001FD02FEFD040400FF0302FE02FFFF000202060001FDFB"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[-1, 2], [6, 1]], [[2, -2], [0, 0]], [[-3, 4], [2, 0]], [[-1, -4], [0, -1]], [[-2, -1], [6, 0]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFFFF07FF00FF030103020100FF03FE020302FD01FE02FC05FD000000060400FF02FCFAFF020000FA00FDFC040502FF0201000000FD00050306FF00FF0400FFFCFCFE000305FD00FFFBFD05000202FE00010601010201FF02000103FFFD00FD0102FA01FD010401FFFFFC050001FE00010200FE00000200FD00000001FFFE01FF0300FE0102FF000101FC03FE0101000703FD00FF02FEFE0101FD0002FF02FF010200FA01FEFF00FFFEFE000000FB04020402FEFA01F7FF02FF010001FD02FEFD040400FF0602FE02FFFF000202060001FDFB"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

