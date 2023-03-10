// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>)
    %2 = call @expected() : () -> tensor<5x6x7xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0x000002FE00FF02020201FEFFFF02FEFFFEFFFD00FD00000100FE020203FF00FF0201FA0203010203010100FF00000001FC0200FFF9FD0305FF000004040200FC0602FD0200FC030000FF0402000002FF02010000040004FC0100FE00010403FE020105FD02FFFFFF010000010102FDFFFD010002FBFE000000FEFBFF030300FEFDFDFD01FD020603FCFF02FE0101F70600FFFFFEFCFDFF00FF0101FF060202FBFF0502FC040206FFFFFC0101FB0200040100FDFD010202FEFFF90000FB0001FEFDFD000000FEFF0301FFFFFAF9000001FF01"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0xFF040100FB01FFFDFB0101FC02FE07000402FFFE00FC0100FE010200FFFD0000FD02FF0401FB05030102FE0306FEFD00FE0103FF020000FEFEFE01F9FF00F90000FBFAFEFE0400FFFCFB0107030100FB020602FD01FE000000FE000300FEFB00000200FDFFFBFEFD00FF0005FF0301FEFE04FA00010304FDFE01FDFA04FD00FE040003000200FF00FE020300"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x0000020000FFFEFAF601FE04FEFCF200F8FE03000000000000FE040003FF00FF0201FA0203010203010100030000000204080005DDF7030A020000F8F40000FC12FEFA000008030000FF0402000002FF02010000F800041CFF000E0000ECEE04FC040003F805FFF9030000FB020CFA03FD010002FBFE000000FEFBFF03030004000000FE000600FA140000FC00FD09E20003000200F10100FFFE01FF060202FBFF0502FC040206FF02F0FA00FB0600F4FE00091204FA0004FC000000F600FF0006FA000000FEFF0301FFFFFAF9000001FF01"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

