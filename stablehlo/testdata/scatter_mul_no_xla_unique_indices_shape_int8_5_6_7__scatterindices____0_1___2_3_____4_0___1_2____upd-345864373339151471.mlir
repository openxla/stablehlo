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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0xFF00FE0400FFFE0000FD01FFFEFF01000100040400FFFDFE010000FEFE02FEFFFE0203030404FFFEFDFF020004FF00FE01FBFF02000001000202060101FE000200FDF9000201000202020201FEFB030100010400000000FFFD02FD01FFFD03000200FF06FD0003FF0106FF020000FDFF0000FFF8FC05FDFC0001050101010500000000FF0001000501FE01FE010403F80201FD00FDFE030000FFFD03FD010500000000FD03FE00FE04FF0000FF00FE02FE020002FE0000FF00000AFF01030204020500FD01FF010403FF0303FEFBFD00FF00"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[4, 1], [4, -7]], [[-4, 4], [3, 1]], [[-2, -5], [0, 3]], [[-2, 5], [-1, 0]], [[-3, 1], [-3, 0]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFF00FE0400FFFE00001501FFFEFF01000100040400FFFDFE010000FEF802FEFFFE0203030404FFFEFDFF020004FF00FE01FBFF02000001000202060401FE000200FDF9000201000202020201FEFB030100010400000000FFFD02FD01FFF703000200FF06FD0003FF0106FF020000FDFF0000FFF8FC05FDFC0001050101010500000000FF0001000001FE01FE010403D80201FD00FDFE030000FF0303FD010500000000FD03FE00FE04030000FF00FE02FE000002FE0000FF00000AFF01030204020500FDFDFF010403FF0303FEFBFD00FF00"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

