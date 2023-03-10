// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>)
    %2 = call @expected() : () -> tensor<5x6x7xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x2xi32>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>) {
    %0 = stablehlo.constant dense<"0x060603010101030002070001040302040201020000010002020303020400020501060101000001000900000002000104010104000100000202060707010202010502020203050102040303020100010004020405010001030401010100040503000403020400000005040102060402020203010200000601050200000301030301030401000203020002010102030003010301050400030001000304020004000502020205050401000001020101010208030100000002010203010005030004010500010403010401000202010203020003"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[0, 0], [0, 6]], [[2, 1], [0, 0]], [[0, 0], [3, 3]], [[5, 1], [1, 0]], [[2, 5], [0, 2]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x0600030101010300022A00010403020402000200000100020203030200000205010601010000010009000000020001040101040001000002020607070102020105020202030500020403030201000100040204050100010304010101000C0503000403020400000005040102060402020603010200000601050200000301030F0103040100020300000201010203000301030105040003000100030402000400050202020505040100000102010101020806010000000201020F010005030004010500010003010401000202010203020003"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

