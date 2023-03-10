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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2xi32>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<2x7xui8>) {
    %0 = stablehlo.constant dense<"0x010000050101000202010003010502000103070601030402040206000004000101030000000201040302040501050001030002010202020401010103020001040202040408030200000500020001000300000600000003000301010000010100030002040305000002050502000104010306050101030006000101010301010600020500010001040303040301030005000002010008030102000000030205030203000003010000010000060505010302010100020202000003040501000000000104000000020002020809030000010005"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[1, 3, 2, 0, 1, 3, 0], [0, 1, 1, 5, 7, 4, 1]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010000050101000203020003030502000103070601030402040206000004000101030000000201040302040501050001030002010202020401010103020001040202040408030200000500020001000300000600000003000301010000010100030002040305000002050502050704010306050101030006000101010301010600020500010001040303040301030005000002010008030102000000030205030203000003010000010000060505010302010100020202000003040501000000000104000000020002020809030000010005"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

