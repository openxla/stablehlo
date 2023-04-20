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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui8>, tensor<2xi32>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>) {
    %0 = stablehlo.constant dense<"0x010203020001000500040100000102000300030202020203000300010103000306010401030204020000020204000104010001050202010005000101000100030A00020202010101000201000302030004030001060001010201010505000002030105060400010400040000010103020004010100000104"> : tensor<4x2x3x5xui8>
    %1 = stablehlo.constant dense<[[2, 1, 0], [4, 0, 0], [2, 7, 0], [0, 1, 1]]> : tensor<4x3xui8>
    return %0, %1 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> tensor<4x2x3x5xui8> {
    %0 = stablehlo.constant dense<"0x010203020201000500040100000102000300030202020203000300010103000306010401030204020000020204000104010001050202010005000101000100030A00020202070101000201000302030004030001060001010201010505000002030105060400010401040000010103020004010100000104"> : tensor<4x2x3x5xui8>
    return %0 : tensor<4x2x3x5xui8>
  }
}

