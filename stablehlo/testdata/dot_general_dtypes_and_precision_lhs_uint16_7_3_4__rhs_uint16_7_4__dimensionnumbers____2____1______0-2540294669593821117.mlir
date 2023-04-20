// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xui16>, tensor<7x4xui16>)
    %1 = call @expected() : () -> tensor<7x3xui16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<7x3x4xui16>, tensor<7x4xui16>) -> tensor<7x3xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xui16>, tensor<7x3xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xui16>, tensor<7x4xui16>) {
    %0 = stablehlo.constant dense<[[[3, 1, 1, 4], [2, 0, 1, 0], [2, 4, 2, 1]], [[0, 2, 5, 0], [0, 0, 1, 0], [1, 0, 0, 0]], [[3, 2, 0, 2], [1, 2, 2, 1], [1, 1, 1, 2]], [[0, 2, 1, 1], [0, 0, 3, 2], [3, 3, 0, 2]], [[3, 1, 5, 0], [1, 3, 2, 7], [0, 2, 1, 0]], [[0, 1, 2, 4], [0, 1, 1, 3], [1, 0, 5, 2]], [[0, 0, 8, 0], [1, 4, 3, 3], [5, 3, 2, 1]]]> : tensor<7x3x4xui16>
    %1 = stablehlo.constant dense<[[0, 0, 1, 2], [0, 0, 2, 0], [4, 2, 0, 3], [2, 2, 4, 6], [1, 1, 2, 1], [3, 3, 4, 0], [0, 4, 5, 3]]> : tensor<7x4xui16>
    return %0, %1 : tensor<7x3x4xui16>, tensor<7x4xui16>
  }
  func.func private @expected() -> tensor<7x3xui16> {
    %0 = stablehlo.constant dense<[[9, 1, 4], [10, 2, 0], [22, 11, 12], [14, 24, 24], [14, 15, 4], [11, 7, 23], [40, 40, 25]]> : tensor<7x3xui16>
    return %0 : tensor<7x3xui16>
  }
}
