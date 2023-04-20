// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xui8>, tensor<1x3xui8>)
    %2 = call @expected() : () -> tensor<1x50x3xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xui8>, tensor<1xi32>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xui8>, tensor<1x3xui8>) {
    %0 = stablehlo.constant dense<"0x030202040004050704030207030102040002010000020301020900000100020301050000010300030306000000040101010101030101030101020301050006010301030202040100050101030302020204000606070300010001040300000101020302020200010101030001000004000301050201030101010204020305020200010101030402010000010103040200000104010300"> : tensor<1x50x3xui8>
    %1 = stablehlo.constant dense<[[3, 0, 1]]> : tensor<1x3xui8>
    return %0, %1 : tensor<1x50x3xui8>, tensor<1x3xui8>
  }
  func.func private @expected() -> tensor<1x50x3xui8> {
    %0 = stablehlo.constant dense<"0x030202040004050704030207030102040002010000020301020900000100020301050000010300030306000000040101010101030101030101020301050006010301030202040100050101030302020204000606070300010001040300000101030302020200010101030001000004000301050201030101010204020305020200010101030402010000010103040200000104010300"> : tensor<1x50x3xui8>
    return %0 : tensor<1x50x3xui8>
  }
}

