// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %2 = call @expected() : () -> tensor<4x2x3x5xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui8>, tensor<2xi32>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>) {
    %0 = stablehlo.constant dense<"0x000202020201000302010401000100010001020001000502050305010201010201010101070202030000030101010302020405000208030400030302050301030004010103000502000004020101010200050202000203010500010000020000030301010005050003000302000000010602040200000202"> : tensor<4x2x3x5xui8>
    %1 = stablehlo.constant dense<[[2, 2, 2], [0, 2, 1], [0, 1, 0], [1, 1, 2]]> : tensor<4x3xui8>
    return %0, %1 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> tensor<4x2x3x5xui8> {
    %0 = stablehlo.constant dense<"0x000202020401000302030401000102010001020001000502050305010201010201010101070202050000030102010302020405000208030400030302050301030004010103010502000004020101010200050202000203010500010000020100030301020005050005000302000000010602040200000202"> : tensor<4x2x3x5xui8>
    return %0 : tensor<4x2x3x5xui8>
  }
}

