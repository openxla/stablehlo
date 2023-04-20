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
    %0 = stablehlo.constant dense<"0x010300030001040103000004000103000003060102010005050202000001020004010103030101020002010004030302050303000302040200000203010005050503050403040600000401020301010302040100030501020004010403000000000008010101040000020105020305030203020101060202000000040100040101020601040003010101000300050003040002030102000101010200000002060400010401030302010000000702000201000406010203010303000202020702010100010403040204030302000004010303"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[2, 5, 3, 0, 5, 0, 1], [1, 3, 0, 3, 3, 6, 1]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010300030001040205030005000103000003060102010005050202000001020004010103030101020002010004030302050303000302040200000203010005050503050403040600000401020301010302040100030501020004010403000000000008010101040000020305030306030203020101060202000000040100040101020601040003010101000300050003040002030102000101010200000002060400010401030302010000000702000201000406010203010303000202020702010100010403040204030302000004010303"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

