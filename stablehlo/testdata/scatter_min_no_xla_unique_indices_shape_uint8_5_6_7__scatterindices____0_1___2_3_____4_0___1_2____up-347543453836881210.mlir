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
    %0 = stablehlo.constant dense<"0x00050006030401010100010101000302000303010000030001000500010400060501000202000500060104030705060100030303070100030102000403040504010302030202060100010105020301000000020100020000010202000103010100000001030102090506060301010101010207000300000402000105000003030402030200040000010202000400030001030A040102040201000301000003010705030301010000000204050008000201000200020401040300010000000000010100020502020105020003020101020100"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[[2, 0], [1, 3]], [[4, 3], [1, 0]], [[1, 2], [0, 2]], [[2, 0], [1, 1]], [[2, 2], [2, 1]]]> : tensor<5x2x2xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x00020006030401010100010101000302000003010000030001000500010400060501000202000500060104030705060100030300070100030102000303040504010302030202010100010105020301000000020100010000010202000102010100000001030102090506060301010101000207000300000402000105000003020402030200040000010202000400030001030A040102040201000101000003010705030301010000000204050008000201000200020401040300010000000000010100020202020105020003020101020100"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

