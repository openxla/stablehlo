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
    %0 = stablehlo.constant dense<"0x050107000005020102000400000103020304010103020004060501000100010301010204000301010700020100050400000002010003000000000502010405010002030001010A03020100020102020304040904030003060001000001000101020201010202010102000007000001000002000103000303060002000500010401040105010103010101010000000102040200010602050101030202000402000202010000030400010107030200000100020300010002000401000402000004040402020103000102020005010000050400"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[1, 0], [0, 1]], [[0, 0], [4, 5]], [[4, 2], [0, 4]], [[0, 5], [3, 0]], [[6, 0], [2, 5]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x050107000005020102010400000103020300010103020004060501000000010301010204000301010700020000050400000002050003000000000500010405010002030001010403020100020102020304040904030403060001000001040101020201010202010102000007000001000002000103000303060002000500010001040105010103000101010000000105040200010602050101030302000402000202010000030400010607030200000100050300010002000400000402000004040402020203000102020005010000050400"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

