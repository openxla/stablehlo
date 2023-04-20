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
      stablehlo.return %arg1 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x2xi32>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>) {
    %0 = stablehlo.constant dense<"0x02000102010005010103000103020001000B010305000103010204000101020400020201030201010202030403010400010102040100020202000302010003020000000400000302030201020102000000000004000103010001000000000302040100030101010401010101050303020606000000000501000102030003040002000000020000000303030100000503070000020002030301010202000001040002000200010101020004010106000400030102000200000103010405030200010002030103010401020000020001010501"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[0, 0], [0, 2]], [[1, 2], [1, 2]], [[6, 6], [1, 2]], [[2, 2], [1, 1]], [[0, 4], [0, 1]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x020001020100050101020001030200010000010305000103010204000001020400020201030201010202030103010400010102020100020202000302010003020000000400000102030201020102000000000004000603010001000000020302040100030106010401010101050303020106000000000501000102030003040202000000020000010303030100000502070000020002030301010102000001040002000200010101020004010106000400010102000200000104010405030200010002030003010401020000020001010501"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

