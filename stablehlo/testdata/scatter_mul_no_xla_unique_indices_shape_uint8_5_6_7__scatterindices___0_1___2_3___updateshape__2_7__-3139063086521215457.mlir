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
    %0 = stablehlo.constant dense<"0x000100020300000101050103010202000003000101020200010100040001000302070000020000010101020000020501000200060001000003050202000000030001040004030001040106000303010002010301000001000102010302000101010006030101040101000100020104050300020002030206010803070101020100030301050000040403010000040403030100070506000100000001000202010304050303000405010302010104000301010006040403020000010004010000000101040005010200030202010201030000"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[5, 1, 4, 0, 2, 5, 0], [3, 3, 0, 1, 5, 1, 6]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x0001000203000005011400060500020000030001010202000101000400010003020700000200000101010200000205010002000600010000030502020000000300010400040300010401060003030100020103010000010001020103020001010100060301010401010003000205041E0300020002030206010803070101020100030301050000040403010000040403030100070506000100000001000202010304050303000405010302010104000301010006040403020000010004010000000101040005010200030202010201030000"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

