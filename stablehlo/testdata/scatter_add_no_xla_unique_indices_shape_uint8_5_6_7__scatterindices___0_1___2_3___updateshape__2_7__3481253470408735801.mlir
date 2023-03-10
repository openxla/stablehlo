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
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2xi32>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<2x7xui8>) {
    %0 = stablehlo.constant dense<"0x010002000204010101030203020202010102030502010002000104060200050201000602010303040A03000101040105010300020200010301000602000204010006000101010600000101010003050301010203030200050101040004020204000201030104000100040300040001010000020002020402050100030700000101000204020005000000000101010201010002020003000003070201030004000103000404030100030001000003010002000103040201010001020601020003050002020201000200010202030300040001"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[0, 4, 0, 4, 6, 1, 0], [4, 6, 3, 3, 0, 1, 0]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010002000204010105030609030202010102030502010002000104060200050201000602010303040A03000101040105010300020200010301000602000204010006000101010600000101010003050301010203030200050101040004020204000201030104000100080903070002010000020002020402050100030700000101000204020005000000000101010201010002020003000003070201030004000103000404030100030001000003010002000103040201010001020601020003050002020201000200010202030300040001"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

