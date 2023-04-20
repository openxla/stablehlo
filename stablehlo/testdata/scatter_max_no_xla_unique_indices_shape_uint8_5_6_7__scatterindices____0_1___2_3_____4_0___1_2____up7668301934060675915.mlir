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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x2xi32>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>) {
    %0 = stablehlo.constant dense<"0x010000010000010101010303030402030104010006000304010201000000040503040203000104010102010301030001030202010400010101010105010005020103010200010201030004000005040601000101040101000100020102000101020701020003030802000100000400030202060001010401000000060202000101020300010103010604000002000205050000030100060401000303020100020004000201030000040101060203000601000102040103000102000202020003000103030401000203010301010307030202"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[2, 3], [4, 0]], [[3, 3], [1, 1]], [[2, 1], [3, 3]], [[7, 0], [6, 1]], [[2, 4], [2, 8]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010200010000010101010303030402030104010006000304010201000400040503040203000104010102010301030001030202010400010101010105010005020103010200010201030004000005040601000101040201000100020102030101020701020003030802000100000400030302060001010401000000060202000701020300010103010604000002000205050000030100060401000603020100020004000201030000040201060203000601080102040103000104000202020003000103030401000203010301010307030202"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

