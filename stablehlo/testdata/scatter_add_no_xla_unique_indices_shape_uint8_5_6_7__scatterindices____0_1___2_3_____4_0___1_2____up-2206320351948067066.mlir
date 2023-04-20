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
    %0 = stablehlo.constant dense<"0x010102000300000001030102010304060105010000010101020504010003000000010602000003020304000207000101010101000301040402020001010100000202000400000006010105000102050000040104000000010002020101020000030301010400020205020003000001050002000102030200010100030200070401010102020301010300000205000105040801010100010406000401020101040000010302010300020202000005020001000402020205010002000103050103010204010202000404020202000202010405"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[0, 2], [3, 1]], [[2, 0], [2, 0]], [[2, 0], [0, 1]], [[5, 0], [1, 1]], [[1, 1], [3, 0]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010102000300000001040102010304060107010000010101020504010303000000010602000003020304000407000101010101000301040402020001010100000202000400000206010105000102050000040104000200010002020101030000030301010400020205020003000001050002000102030200010100030200070901010102020301020300000205000105040801010100010406000501020101040000010302010300020302000005020001000402020205010003000103050103010204010502000404020202000202010405"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

