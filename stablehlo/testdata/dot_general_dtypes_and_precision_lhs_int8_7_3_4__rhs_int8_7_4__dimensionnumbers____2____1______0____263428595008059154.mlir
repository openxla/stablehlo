// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xi8>, tensor<7x4xi8>)
    %1 = call @expected() : () -> tensor<7x3xi8>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<7x3x4xi8>, tensor<7x4xi8>) -> tensor<7x3xi8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xi8>, tensor<7x3xi8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xi8>, tensor<7x4xi8>) {
    %0 = stablehlo.constant dense<[[[0, 5, 2, 0], [4, -2, 0, -2], [1, -1, -3, -4]], [[6, 2, -1, -4], [0, 2, 0, 0], [3, -1, 3, -1]], [[-1, 2, 3, 1], [0, -3, 3, 2], [3, -3, 0, -1]], [[0, 1, 0, -3], [0, 4, 1, -1], [0, 1, 0, -1]], [[3, -2, -3, 3], [-2, -1, 0, -3], [-1, 4, 1, 0]], [[-1, -1, -3, -2], [-5, 1, 0, 2], [0, 1, -4, -2]], [[0, -1, 2, 1], [0, -1, 8, -6], [2, -2, 6, 2]]]> : tensor<7x3x4xi8>
    %1 = stablehlo.constant dense<[[0, 1, -1, -2], [-3, 0, 1, 0], [3, 1, 2, 0], [0, -2, 0, 3], [-2, 1, 3, 6], [0, 0, 2, 9], [-2, 3, 2, -3]]> : tensor<7x4xi8>
    return %0, %1 : tensor<7x3x4xi8>, tensor<7x4xi8>
  }
  func.func private @expected() -> tensor<7x3xi8> {
    %0 = stablehlo.constant dense<[[3, 2, 10], [-19, 0, -6], [5, 3, 6], [-11, -11, -5], [1, -15, 9], [-24, 18, -26], [-2, 31, -4]]> : tensor<7x3xi8>
    return %0 : tensor<7x3xi8>
  }
}
