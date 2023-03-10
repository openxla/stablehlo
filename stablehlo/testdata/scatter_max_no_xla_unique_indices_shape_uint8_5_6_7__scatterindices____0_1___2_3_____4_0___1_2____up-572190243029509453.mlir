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
    %0 = stablehlo.constant dense<"0x010102010100050201030100000106080305010102010004010302050000010401020004040200020000030102010002030403030400010603010102010300030003030304040409000301010200010303010205010302010100040000010200020304030109010105030103030401040203010001030204000002010002030100010003030001000201000901000000000201000202000001020200010101040100040400000208030104010001050300040000020601000101010202020400050405010200030400000103040204020201"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[0, 5], [0, 4]], [[2, 4], [0, 2]], [[1, 1], [0, 0]], [[2, 4], [1, 5]], [[3, 0], [0, 0]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010102010100050201040100000106080305010102010004010302050000010401020004040200020000030202010002030403030400010603010104010300030003030304040409000301010200010303010205010302010100040000010200020304030109010105030103030401040203010001030204000002010002030200010003030001050201000901000004000201000202000001020200010101040100040400000208030304010001050300040000020601000101010202020400050405010200030400000103040204020201"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

