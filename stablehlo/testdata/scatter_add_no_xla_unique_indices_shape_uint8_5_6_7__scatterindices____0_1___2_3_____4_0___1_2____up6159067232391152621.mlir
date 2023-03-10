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
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x2xi32>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>) {
    %0 = stablehlo.constant dense<"0x020201020400040301020004020100020400010106010201000000030100000201040004000500000201000001050004050601000000040000020000000100010400020004010202000304000103020100040300020503010407030403020101000205010001050600010006010000000501030802050100070002030300030400010100000002060301030001030305010101000002000102010108030004000202040301020200020000050100020003010703040001020504010200020201010204010100040101000400000203060104"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[1, 5], [2, 1]], [[0, 0], [2, 0]], [[2, 2], [8, 5]], [[1, 5], [2, 3]], [[0, 7], [0, 1]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x020301020400040301030004020100020405010106010201000000030300000201040004000500000201000001050004050601000000040000020000000100010400020004010402000304000103020100040300020703010407030403070101000205010003050600010006010000000D0103080205010007000203030003050001010000000209030103000103030A01010100000200010201030803000400020204030102020002000005010002000302070304000102050B010200020201010204010100040101000400000203060104"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

