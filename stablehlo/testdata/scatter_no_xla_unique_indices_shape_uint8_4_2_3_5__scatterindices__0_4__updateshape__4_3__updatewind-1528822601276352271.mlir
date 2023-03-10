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
      stablehlo.return %arg1 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xui8>, tensor<2xi32>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>) {
    %0 = stablehlo.constant dense<"0x030104000301050603020603010106000300010201010201010301010201010801030200000000010100060300040000060200020004050204020005020203010303000400000104060200010300040004020302050200000103010002000200050205030000010201030301070000010001010203000400"> : tensor<4x2x3x5xui8>
    %1 = stablehlo.constant dense<[[2, 0, 3], [3, 4, 0], [1, 0, 3], [1, 0, 3]]> : tensor<4x3xui8>
    return %0, %1 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> tensor<4x2x3x5xui8> {
    %0 = stablehlo.constant dense<"0x030104000201050603000603010103000300010201010201010301010201010801030300000000040100060300040000060200020004050204020005020203010103000400000104060203010300040004020302050200000103010002000100050205000000010203030301070000010001010203000400"> : tensor<4x2x3x5xui8>
    return %0 : tensor<4x2x3x5xui8>
  }
}

