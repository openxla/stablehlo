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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0xFFFFFE04FEFB01F9000000000101FFFC02FBFB0303FE02FC0004010302FF000103FD02FEFEFEFE0500FF0002FFFF0504FDFF0100050000000001FFFE02000103FA0001FEFCFCFFFE01000202FFFF0000FF06FF02020100FF01FCFDFE02FEFB01010400FDFE02FF01FF03FE0300FDFCFE000000FBFFFF00FE0301FE00FEFF00040002060200FCFA0000FDFF02030000FCFEFDFD01FEFF00FE04F902FEFEFE000201FE000100000001FC0002FDFD04FCFF000000FDFE030202FF00FF03020503FFFDFC020300FD0004FFFF0301FFFDFA00FD00"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[-4, 1], [-4, 2]], [[-1, 1], [0, -4]], [[1, 2], [-1, -1]], [[-1, 0], [-1, -1]], [[3, -3], [0, 0]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFFFCFE04FEFB01F9000000000101FFFC02FBFB0303FE02FC00040103FCFF000103FD02FEFEFEFE0500FF00FFFFFF0504FDFF01FC050000000001FFFE02000103FA0001FEFCFCFFFE01000202FFFF0000FF06FF02020100FF01FCFDFE02FEFB01010400FDFE02FF01FF03FE0300FDFCFEFF0000FBFFFF00FE0301FE00FEFF00FF0002060200FCFAFF00FDFF02030000FCFEFDFD01FEFF00FE04F9FFFEFEFE000201FE000100000001FC0002FDFD04FCFF000000FDFE030202FFFDFF03020503FFFDFC020300FD0004FFFF0301FFFDFA00FD00"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

