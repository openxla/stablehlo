// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xi32>, tensor<7x4xi32>)
    %1 = call @expected() : () -> tensor<7x3xi32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<7x3x4xi32>, tensor<7x4xi32>) -> tensor<7x3xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xi32>, tensor<7x3xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xi32>, tensor<7x4xi32>) {
    %0 = stablehlo.constant dense<[[[0, 0, 3, 0], [1, -3, 0, 4], [-5, 4, -3, 7]], [[0, 1, 5, 0], [3, 0, -1, 6], [0, -2, -1, 1]], [[0, 0, 0, 3], [-3, 8, 1, 0], [-1, -3, -1, -1]], [[1, 5, 0, 0], [5, -2, -7, -3], [-2, -3, -3, 2]], [[0, 0, -4, 0], [1, -1, 5, -4], [0, 2, -5, 0]], [[1, 0, 0, 3], [-1, -4, -6, 1], [0, 0, -1, -1]], [[0, 2, 0, 5], [1, 6, 4, -1], [-1, -6, -6, 1]]]> : tensor<7x3x4xi32>
    %1 = stablehlo.constant dense<[[1, 3, 4, -8], [0, 0, 0, 4], [3, -3, 4, 2], [-3, 0, -2, 2], [0, -2, 5, 1], [-2, 4, 1, 4], [-1, 0, 2, 2]]> : tensor<7x4xi32>
    return %0, %1 : tensor<7x3x4xi32>, tensor<7x4xi32>
  }
  func.func private @expected() -> tensor<7x3xi32> {
    %0 = stablehlo.constant dense<[[12, -40, -61], [0, 24, 4], [6, -29, 0], [-3, -7, 16], [-20, 23, -29], [10, -16, -5], [10, 5, -9]]> : tensor<7x3xi32>
    return %0 : tensor<7x3xi32>
  }
}

