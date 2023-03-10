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
      %5 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi8>, tensor<2x2x2xi32>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>) {
    %0 = stablehlo.constant dense<"0x0000FDFD02FE0500FC00FE010000FD02FF0402FB07010200FD0202FFFB0004FE030005FF0001FEF9FEFCF7FE0101010002FEFE0201040101FC000100FFFF01FF04FFFD010100FE000000FA0201020400FCFA02FDFBFDFEFDFF0202FE00FFFF00FF010208FFFE000000FCFCFEFAFD03FE01FD05FEFC05FFFF00040604FA0BFF01FF000503FA00FFFEFD0000FBFB0003FD040201000302050100FEFE02FC05FD02FFFC0002FFFF0003FEFF02FCFF00FF02FC0000010001030103010003000403FF00020507010302FB0101000301FD0204FE01"> : tensor<5x6x7xi8>
    %1 = stablehlo.constant dense<[[[-2, 1], [4, 2]], [[-1, 0], [4, 2]], [[0, -2], [0, -1]], [[0, 0], [6, 0]], [[0, 1], [0, -2]]]> : tensor<5x2x2xi8>
    return %0, %1 : tensor<5x6x7xi8>, tensor<5x2x2xi8>
  }
  func.func private @expected() -> tensor<5x6x7xi8> {
    %0 = stablehlo.constant dense<"0x00FEFDFD02FE0500FC02FE010000FD02FF0502FB07010200FD0202FFFF0004FE030005FF0001FEF9FEFCF7FD0101010002FEFE0401040101FC000100FFFF01FF04FFFD01010002000000FA0201020400FCFA02FDFBFDFEFDFF0202FE00FEFF00FF010208FFFC000000FCFCFEFAFD03FE01FD05FEFC05FFFF00040604FA0BFF01FF000503FA00FFFEFD0000FBFB0003FD040201000302050100FE0402FC05FD02FFFC0002FFFF0003FEFF02FCFF00FF02FCFE00010001030103020003000403FF00020507010302FB0101000301FD0204FE01"> : tensor<5x6x7xi8>
    return %0 : tensor<5x6x7xi8>
  }
}

