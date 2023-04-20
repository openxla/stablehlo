// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x2x3xui8>, tensor<2x3xui8>)
    %2 = call @expected() : () -> tensor<1x2x3xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<1x2x3xui8>, tensor<1xi32>, tensor<2x3xui8>) -> tensor<1x2x3xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x2x3xui8>, tensor<1x2x3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2x3xui8>, tensor<2x3xui8>) {
    %0 = stablehlo.constant dense<[[[0, 2, 0], [3, 1, 5]]]> : tensor<1x2x3xui8>
    %1 = stablehlo.constant dense<[[0, 1, 3], [1, 8, 5]]> : tensor<2x3xui8>
    return %0, %1 : tensor<1x2x3xui8>, tensor<2x3xui8>
  }
  func.func private @expected() -> tensor<1x2x3xui8> {
    %0 = stablehlo.constant dense<[[[0, 2, 0], [3, 8, 25]]]> : tensor<1x2x3xui8>
    return %0 : tensor<1x2x3xui8>
  }
}

