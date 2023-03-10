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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2xi32>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<2x7xui8>) {
    %0 = stablehlo.constant dense<"0x010004000101030203010202000004020000010201000003010200020303010006010004000004000106000503050104000600020500010201050101010302040102010100030401000000000101000201030000020008000101000002010202070005040101000200000001020202010101030405040202030304030004010103000007040100010404010300020201010306000100000002000301020301010004040404000105000000040503000000050103040100050201020300000304030403030000000202000304050205000004"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[0, 3, 0, 1, 0, 2, 3], [0, 1, 1, 3, 4, 0, 3]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x010004000101030009000200000004020000010201000003010200020303010006010004000004000106000503050104000600020500010201050101010302040102010100030401000000000101000201030000020008000101000002010202070005040101000200000001060800030101030405040202030304030004010103000007040100010404010300020201010306000100000002000301020301010004040404000105000000040503000000050103040100050201020300000304030403030000000202000304050205000004"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

