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
    %0 = stablehlo.constant dense<"0x000301030000010001000102000001020102020104040102020205080206030401000705010304000101020001020101010202000001020100050300000302040301040301010204020002010200010003000207040201030004000301000001020001030101020500000102050002020205000000040300030401000001030001000204000201030206020004000004010100020302010001020303010403000300040300010100010001040001040202030400020201000303020400020003020200050401020003020000060300010101"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[0, 5], [0, 0]], [[5, 0], [2, 0]], [[4, 0], [6, 0]], [[0, 0], [2, 1]], [[0, 1], [3, 0]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x00000103000001000100010200000102010A020104040102020205080006030401000705010304000101020001020101010202000001020100050300000302040301040301010404020002010200010003000207040801030004000301000001020001030100020500000102050002020C05000000040300030401000001030001000204000201030206020004000000010100020302010001020603010403000300040300010100010001040001040202000400020201000303020400020003020200050C01020003020000060300010101"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

