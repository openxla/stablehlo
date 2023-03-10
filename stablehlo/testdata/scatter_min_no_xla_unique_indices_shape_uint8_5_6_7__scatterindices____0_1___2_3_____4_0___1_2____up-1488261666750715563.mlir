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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x2xi32>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>) {
    %0 = stablehlo.constant dense<"0x010100020402010301000200000203010105060000010100020404020102010100020101000502000101030002030200010103000300000001010300000203040102040202010000000604010400000001040300030201030101010103020500010600000100010001030401020000010204000101060602030003090500010003000200030005010200000201080205010601030001000001030004050202000100020300030004010200010001010607000002000101030105000203010004030100000100020600060101030402000200"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[3, 2], [3, 4]], [[0, 1], [0, 3]], [[1, 5], [3, 0]], [[0, 3], [2, 0]], [[2, 5], [0, 5]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010100020402010301000200000203010102060000010100020404020102010100020101000502000101030002030200010103000300000001010300000203040102040202010000000604010400000001040300030101030101010103000500010600000100010001030401020000010204000101060602030003090500010003000200030005000200000201080203010601030001000001030004050202000100020300030004010200010001010607000002000101030105000203010004030100000000020600060101030402000200"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

