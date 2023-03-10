// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xi32>, tensor<7x4xi32>)
    %1 = call @expected() : () -> tensor<7x3xi32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<7x3x4xi32>, tensor<7x4xi32>) -> tensor<7x3xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xi32>, tensor<7x3xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xi32>, tensor<7x4xi32>) {
    %0 = stablehlo.constant dense<[[[0, 0, 0, -3], [-2, -3, 2, 4], [-3, -1, 1, 4]], [[3, 0, 2, 1], [0, 2, -4, 1], [0, 2, 1, 1]], [[0, -2, 0, 4], [1, 0, 3, -4], [0, -2, -1, 0]], [[4, 5, 2, 2], [0, 2, 5, -2], [2, 2, -3, 0]], [[-1, 2, 6, -2], [3, -2, 4, -2], [2, -3, 0, 5]], [[2, -5, 9, 3], [1, 8, -1, 0], [0, 0, -3, 0]], [[-4, -1, -2, 2], [1, 1, -2, 2], [2, 0, 3, 1]]]> : tensor<7x3x4xi32>
    %1 = stablehlo.constant dense<[[-1, -2, 0, -1], [2, 1, 0, 0], [3, 0, 0, -1], [2, 3, 3, -3], [1, 6, 3, 0], [-4, 3, 0, 0], [1, -4, -3, 5]]> : tensor<7x4xi32>
    return %0, %1 : tensor<7x3x4xi32>, tensor<7x4xi32>
  }
  func.func private @expected() -> tensor<7x3xi32> {
    %0 = stablehlo.constant dense<[[3, 4, 1], [6, 2, 2], [-4, 7, 0], [23, 27, 1], [29, 3, -16], [-23, 20, 0], [16, 13, -2]]> : tensor<7x3xi32>
    return %0 : tensor<7x3xi32>
  }
}
