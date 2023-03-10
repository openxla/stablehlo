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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x1xi32>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) {
    %0 = stablehlo.constant dense<"0x00FE05FFFB0201020400000300030200FD0203FE000102FF0101020301FE02FB0401FF0004020000FC05FD0404FE07FBFB0000000000FFFB00FC0301040001060001FE000002000204FFFF0000FB00FD00FF00FF01FDFDFC00FEFE010000020003FF00FD020300020100020600FF02FD01FFFE0003FCFEFB010000010203FF0800010002FDFD00FD00FD030003FEFEFF0100FF05FF020300030000FF0000FE0401FF00FFFF02FC030000FD000004020000FF01FD0204020004FD0201FDFDFF020002FE00FE000000000102FD0101FF050501"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<"0x0001FD0300FE00FE0101FEFFFF0000020000FF00FF040A06F9FE00FF03FE0000FDFB00FE0002FF000504040A01000103000000000000FF010007FF01010002FD03000203FF02000002FAFD0300FAFF03000000FF0300FEFFFE000004FC00040202FEFFFD00030003FEFD0000FF030002FFFFFC0100FDFF00000102040201FEFD000302FEFC0001000302FF02"> : tensor<5x2x2x7xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2x7xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x00FEFDFFFBFE00FE0100FEFFFF000000FD00FFFEFF0102FFF9FE00FF01FE02FB0401FF0004020000FC05FDFE00FEFDFBFBFE0000FF00FFFB00FC0100010000000000FE00FF01000204FFFF0000FB00FD00FF00FF00FDFDFC00FEFEFD00000200FFFF00FD02FAFD0200FAFF0300FF00FD01FFFE0003FCFEFB010000010203FF00FEFFFE00FDFDFCFD00FD02FEFFFDFEFF0000FEFDFF00FF00000000FF0000FE0401FF00FFFF02FC03FFFFFC0000FDFF0000FF01FD0201FEFD00FD02FEFCFDFF000002FE00FE000000000102FD0101FF050501"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

