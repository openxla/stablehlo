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
    %0 = stablehlo.constant dense<"0xFD0500FEFF04020002FF00FF0002000002020805FB03FE01FFFC00000102FE030001FD00FBFF00FC0104FEFC0203FBFFFAFEFF00FF0001000004010201FFFFFF00FD010203FCFE02000306020402FF020000FD01FFFAFF0000FFFF06FDFF0100FF0000000000FF00FFFE03000AFC0100FD000300FEFF0003FD0000020002FE00FD0301FF00040101FB05FD02FF01FD030102FB0101020203FCFE02FE010301000300FE00FD0402FC0301FB0001010100FE00000000FE01FF01FD00FEFDF80000FEFC00030002FFFFFE02FE0205020500FE00"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[0, 7, 6, 2, 0, 0, -1], [-4, 1, -2, 3, 0, 0, 0]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0xFD0500FEFF040200070602000002000002020805FB03FE01FFFC00000102FE030001FD00FBFF00FC0104FEFC0203FBFFFAFEFF00FF0001000004010201FFFFFF00FD010203FCFE02000306020402FF020000FD01FFFAFF0000FFFF06FDFF0100FF0000000000FF00FFFE03000A000100FD000300FEFF0003FD0000020002FE00FD0301FF00040101FB05FD02FF01FD030102FB0101020203FCFE02FE010301000300FE00FD0402FC0301FB0001010100FE00000000FE01FF01FD00FEFDF80000FEFC00030002FFFFFE02FE0205020500FE00"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

