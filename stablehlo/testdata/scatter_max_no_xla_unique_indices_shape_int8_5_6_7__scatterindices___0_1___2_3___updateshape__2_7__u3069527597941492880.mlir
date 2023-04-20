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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2xi32>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<2x7xi8>) {
    %0 = stablehlo.constant dense<"0x0407FFFEFB0002FDFF00FE010000030501FB030000FF000000F9FFFFFD0007040205FD000101FF00FF0000FBFD00010300FEFC000006FD03030100000300FE020000010106F70100FE00FF0004FD0001FF0400FEFF0400FE000200FCFF04FE06FEFE04FD000001000001000300FE0000FFFC01FF000100070000FF0300020000FFFF01FF0207FD05FFFD0000FF0403000002FF000000FF0100FEFF0100FFFFFDFE0100FC05FC040003000002FEFC02030002FD05FF0004FFFD02020402FFFE00FF0000FC010202FEFAFF00010000FF02FE03"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[-1, 3, -5, 3, 3, -1, 2], [-2, 0, 1, -4, -3, 1, -3]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x0407FFFEFB0002FF030003030002030501FB030000FF000000F9FFFFFD0007040205FD000101FF00FF0000FBFD00010300FEFC000006FD03030100000300FE020000010106F70100FE00FF0004FD0001FF0400FEFF0400FE000200FCFF04FE06FEFE04FD000001000001000300FE0100FFFC01FF000100070000FF0300020000FFFF01FF0207FD05FFFD0000FF0403000002FF000000FF0100FEFF0100FFFFFDFE0100FC05FC040003000002FEFC02030002FD05FF0004FFFD02020402FFFE00FF0000FC010202FEFAFF00010000FF02FE03"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

