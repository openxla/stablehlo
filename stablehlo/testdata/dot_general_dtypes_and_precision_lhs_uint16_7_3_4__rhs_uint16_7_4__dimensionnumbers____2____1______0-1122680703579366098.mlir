// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xui16>, tensor<7x4xui16>)
    %1 = call @expected() : () -> tensor<7x3xui16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<7x3x4xui16>, tensor<7x4xui16>) -> tensor<7x3xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xui16>, tensor<7x3xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xui16>, tensor<7x4xui16>) {
    %0 = stablehlo.constant dense<[[[1, 1, 3, 2], [5, 0, 1, 1], [3, 0, 3, 4]], [[0, 3, 5, 0], [1, 4, 0, 3], [6, 4, 1, 8]], [[0, 1, 2, 1], [1, 0, 1, 1], [4, 2, 5, 2]], [[1, 1, 6, 5], [3, 2, 1, 1], [1, 0, 3, 4]], [[0, 3, 5, 1], [6, 5, 2, 4], [3, 5, 0, 2]], [[3, 2, 1, 0], [3, 3, 4, 0], [2, 2, 0, 4]], [[2, 1, 4, 0], [0, 3, 1, 0], [1, 5, 0, 1]]]> : tensor<7x3x4xui16>
    %1 = stablehlo.constant dense<[[3, 0, 0, 2], [1, 0, 2, 1], [3, 4, 3, 1], [1, 5, 3, 3], [2, 0, 0, 3], [4, 1, 2, 1], [2, 2, 3, 2]]> : tensor<7x4xui16>
    return %0, %1 : tensor<7x3x4xui16>, tensor<7x4xui16>
  }
  func.func private @expected() -> tensor<7x3xui16> {
    %0 = stablehlo.constant dense<[[7, 17, 17], [10, 4, 16], [11, 7, 37], [39, 19, 22], [3, 24, 12], [16, 23, 14], [18, 9, 14]]> : tensor<7x3xui16>
    return %0 : tensor<7x3xui16>
  }
}
