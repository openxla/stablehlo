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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2xi32>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<2x7xui8>) {
    %0 = stablehlo.constant dense<"0x050102040001070102060500000102000202010201020102030203040300000203010203000005030201010302020101030000020300050103000900050403000102020105010000040105040001050107000005020001060100000104010000000403000104020002010402030202010001040501080100010103000301030203020303000505020000030002040502020100040400020006030002040101000400030400000000030102040300020001000001000603020302000401010102010403000100010104050000010000000302"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<[[4, 2, 0, 2, 3, 1, 7], [3, 4, 0, 4, 0, 2, 1]]> : tensor<2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x050102040001070102000200000102000202010201020102030203040300000203010203000005030201010302020101030000020300050103000900050403000102020105010000040105040001050107000005020001060100000104010000000403000104020002010400030002010001040501080100010103000301030203020303000505020000030002040502020100040400020006030002040101000400030400000000030102040300020001000001000603020302000401010102010403000100010104050000010000000302"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

