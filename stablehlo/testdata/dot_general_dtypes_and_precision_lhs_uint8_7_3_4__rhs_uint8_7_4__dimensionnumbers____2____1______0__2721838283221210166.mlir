// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xui8>, tensor<7x4xui8>)
    %1 = call @expected() : () -> tensor<7x3xui8>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<7x3x4xui8>, tensor<7x4xui8>) -> tensor<7x3xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xui8>, tensor<7x3xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xui8>, tensor<7x4xui8>) {
    %0 = stablehlo.constant dense<[[[4, 1, 1, 6], [4, 1, 0, 0], [2, 1, 0, 1]], [[2, 0, 0, 2], [1, 1, 3, 6], [4, 1, 3, 3]], [[0, 0, 3, 2], [0, 3, 0, 0], [0, 2, 4, 4]], [[3, 3, 1, 4], [1, 8, 0, 1], [1, 0, 1, 1]], [[2, 7, 2, 3], [0, 2, 0, 3], [1, 3, 3, 7]], [[1, 1, 1, 1], [1, 6, 0, 1], [3, 2, 0, 0]], [[1, 0, 2, 0], [6, 1, 0, 4], [1, 0, 5, 1]]]> : tensor<7x3x4xui8>
    %1 = stablehlo.constant dense<[[1, 2, 0, 0], [0, 0, 0, 3], [1, 2, 5, 1], [2, 0, 0, 0], [0, 0, 2, 6], [1, 1, 5, 3], [3, 1, 3, 0]]> : tensor<7x4xui8>
    return %0, %1 : tensor<7x3x4xui8>, tensor<7x4xui8>
  }
  func.func private @expected() -> tensor<7x3xui8> {
    %0 = stablehlo.constant dense<[[6, 6, 4], [6, 18, 9], [17, 6, 28], [6, 2, 2], [22, 18, 48], [10, 10, 5], [9, 19, 18]]> : tensor<7x3xui8>
    return %0 : tensor<7x3xui8>
  }
}

