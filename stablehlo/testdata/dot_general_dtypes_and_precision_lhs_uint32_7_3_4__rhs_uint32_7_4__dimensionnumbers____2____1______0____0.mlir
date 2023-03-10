// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xui32>, tensor<7x4xui32>)
    %1 = call @expected() : () -> tensor<7x3xui32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<7x3x4xui32>, tensor<7x4xui32>) -> tensor<7x3xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xui32>, tensor<7x3xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xui32>, tensor<7x4xui32>) {
    %0 = stablehlo.constant dense<[[[4, 0, 1, 0], [1, 7, 0, 1], [0, 2, 6, 1]], [[2, 0, 0, 1], [8, 2, 1, 4], [2, 1, 2, 3]], [[1, 2, 0, 7], [3, 0, 1, 5], [1, 0, 1, 1]], [[0, 1, 5, 1], [3, 2, 4, 2], [5, 1, 7, 1]], [[2, 4, 5, 4], [4, 2, 1, 4], [5, 3, 1, 0]], [[2, 3, 1, 1], [0, 1, 2, 1], [2, 1, 3, 2]], [[2, 5, 1, 4], [5, 2, 1, 3], [2, 3, 3, 3]]]> : tensor<7x3x4xui32>
    %1 = stablehlo.constant dense<[[0, 4, 4, 3], [7, 2, 1, 1], [0, 4, 2, 1], [0, 0, 2, 2], [3, 0, 1, 3], [3, 2, 1, 1], [0, 2, 2, 0]]> : tensor<7x4xui32>
    return %0, %1 : tensor<7x3x4xui32>, tensor<7x4xui32>
  }
  func.func private @expected() -> tensor<7x3xui32> {
    %0 = stablehlo.constant dense<[[4, 31, 35], [15, 65, 21], [15, 7, 3], [12, 12, 16], [23, 25, 16], [14, 5, 13], [12, 6, 12]]> : tensor<7x3xui32>
    return %0 : tensor<7x3xui32>
  }
}

