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
    %0 = stablehlo.constant dense<"0x020006000102010202000000020103000005000303010001010504000001050100020100010000040100050304050102020201020000020001050102050001020200030300000300010203000002030301040503030301000102000101030002040001020103040102030400010201010500020201010002060002010401020202020201040101010000020101020307010103030001020100010401060000050100020300010101010001010201010100000201000100020000000101000202000500000105000100020103000100010200"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[3, 3, 1, 3, 1, 1, 1], [0, 0, 0, 1, 1, 3, 1]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x020006000102010505010301030203000005000303010001010504000001050100020100010000040100050304050102020201020000020001050102050001020200030300000300010203000002030301040503030301000102000101030002040001020103040102030400020304020500020201010002060002010401020202020201040101010000020101020307010103030001020100010401060000050100020300010101010001010201010100000201000100020000000101000202000500000105000100020103000100010200"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

