// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<2x7xui8>)
    %2 = call @expected() : () -> tensor<5x6x7xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      stablehlo.return %arg1 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2xi32>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<2x7xui8>) {
    %0 = stablehlo.constant dense<"0x000002030302020206030001010101010104020104010500020000040200040000040302040106000202010001030200010102000000000203000601020002010000020207030303040205010200030001010102050302040002070100020103020503000301010400030002020604000300000402010500010303020100000202030001010002010203010000000002020201000304030103030001010000010604010301010201010000020002030000000203000001000204080101020004010000010205000405010501000103010405"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[4, 0, 0, 0, 2, 3, 0], [3, 2, 3, 2, 3, 0, 0]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x000002030302020400000002030001010104020104010500020000040200040000040302040106000202010001030200010102000000000203000601020002010000020207030303040205010200030001010102050302040002070100020103020503000301010400030203020300000300000402010500010303020100000202030001010002010203010000000002020201000304030103030001010000010604010301010201010000020002030000000203000001000204080101020004010000010205000405010501000103010405"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

