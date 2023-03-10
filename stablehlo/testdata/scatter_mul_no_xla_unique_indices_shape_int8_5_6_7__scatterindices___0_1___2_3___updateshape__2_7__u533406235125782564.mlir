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
    %0 = stablehlo.constant dense<"0x01000000FE00000403FD0001FEF901FD00FC00FB0701FE0004FFFC00FFFFFD000302FE04FDFB01FE0400030100FEFBFFFEFFFBFEFF030000FB0102030002FF030200FE0000FFFD0101FE010203FB000304FD00FF0101FE0104FC01FA0000FF0400010004FF02000200FEFF01FCFE05FE00FFFD0301FFFEFF01FD0003FE00000100050105FFFF000301FD000004FC0200FF0002FEF900000104060005FEFFFB0301FCFFFEFF0204FD0401FD0000FE02040000FBFF00FE0300FE00020000000403FE00010002FB00FF0203FEFFFFFD00FE0003"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[4, -2, -6, -2, -1, 2, 0], [-1, 0, 2, 3, -5, 0, -2]]> : tensor<2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x01000000FE000010FA1200FFFC0001FD00FC00FB0701FE0004FFFC00FFFFFD000302FE04FDFB01FE0400030100FEFBFFFEFFFBFEFF030000FB0102030002FF030200FE0000FFFD0101FE010203FB000304FD00FF0101FE0104FC01FA0000FF0400010004FF02000200020002F40A000400FFFD0301FFFEFF01FD0003FE00000100050105FFFF000301FD000004FC0200FF0002FEF900000104060005FEFFFB0301FCFFFEFF0204FD0401FD0000FE02040000FBFF00FE0300FE00020000000403FE00010002FB00FF0203FEFFFFFD00FE0003"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

