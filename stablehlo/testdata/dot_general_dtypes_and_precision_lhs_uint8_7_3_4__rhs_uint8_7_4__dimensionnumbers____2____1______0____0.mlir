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
    %0 = stablehlo.constant dense<[[[1, 2, 4, 1], [1, 0, 0, 0], [1, 1, 1, 3]], [[1, 7, 1, 2], [2, 2, 0, 2], [3, 5, 2, 2]], [[0, 5, 2, 0], [1, 3, 0, 1], [3, 4, 3, 3]], [[4, 2, 1, 1], [3, 1, 1, 3], [3, 2, 0, 3]], [[2, 0, 1, 3], [6, 4, 0, 1], [0, 3, 1, 1]], [[1, 2, 2, 0], [2, 5, 1, 0], [0, 0, 6, 2]], [[5, 3, 1, 0], [2, 1, 0, 6], [3, 7, 1, 3]]]> : tensor<7x3x4xui8>
    %1 = stablehlo.constant dense<[[1, 2, 1, 2], [5, 4, 2, 0], [0, 2, 2, 0], [1, 3, 0, 6], [2, 3, 2, 5], [2, 2, 3, 1], [1, 1, 0, 2]]> : tensor<7x4xui8>
    return %0, %1 : tensor<7x3x4xui8>, tensor<7x4xui8>
  }
  func.func private @expected() -> tensor<7x3xui8> {
    %0 = stablehlo.constant dense<[[11, 1, 10], [35, 18, 39], [14, 6, 14], [16, 24, 27], [21, 29, 16], [12, 17, 20], [8, 15, 16]]> : tensor<7x3xui8>
    return %0 : tensor<7x3xui8>
  }
}

