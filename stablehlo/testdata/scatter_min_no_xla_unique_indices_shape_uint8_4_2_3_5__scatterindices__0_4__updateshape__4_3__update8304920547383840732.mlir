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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui8>, tensor<2xi32>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>) {
    %0 = stablehlo.constant dense<"0x050102030002030000000500010002000002000600010102050202050803020003020201010101020004050202000104000300030503010103010302030301000101050000000003070007040201010004000502000001000101040500030602020503010401050302020701070201010201020502060400"> : tensor<4x2x3x5xui8>
    %1 = stablehlo.constant dense<[[5, 0, 0], [2, 1, 4], [0, 2, 3], [3, 3, 0]]> : tensor<4x3xui8>
    return %0, %1 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> tensor<4x2x3x5xui8> {
    %0 = stablehlo.constant dense<"0x050102030002030000000500010000000002000600010102050202050803020003020201010101010004050202000104000300030503010103010302030301000001050000000003070003040201010004000502000001000101040500030302020503010401050300020701070201010201020502060400"> : tensor<4x2x3x5xui8>
    return %0 : tensor<4x2x3x5xui8>
  }
}

