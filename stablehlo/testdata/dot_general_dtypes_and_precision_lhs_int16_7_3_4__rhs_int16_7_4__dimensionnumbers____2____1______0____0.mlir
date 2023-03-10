// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xi16>, tensor<7x4xi16>)
    %1 = call @expected() : () -> tensor<7x3xi16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<7x3x4xi16>, tensor<7x4xi16>) -> tensor<7x3xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xi16>, tensor<7x3xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xi16>, tensor<7x4xi16>) {
    %0 = stablehlo.constant dense<[[[3, 1, -2, 2], [0, -3, -8, 5], [1, -1, 1, 0]], [[0, 1, -1, 3], [0, 0, 3, 2], [1, 0, -3, 3]], [[3, 3, -1, -1], [-4, 0, 0, 1], [3, 0, 2, 2]], [[0, -1, 5, 0], [0, 0, 0, -1], [1, 8, 4, 3]], [[0, -4, 0, 2], [1, -6, -2, 0], [0, 1, 3, 0]], [[4, 1, 6, 4], [-2, -1, -3, 3], [4, -2, -2, 1]], [[0, 0, -8, 0], [1, -1, -1, 4], [3, -3, 0, 1]]]> : tensor<7x3x4xi16>
    %1 = stablehlo.constant dense<[[1, 1, -1, 0], [-1, 3, -1, -6], [-1, 2, 4, 1], [0, 4, -3, 4], [1, -5, 6, 1], [-1, -2, 3, 0], [-2, -2, 2, 0]]> : tensor<7x4xi16>
    return %0, %1 : tensor<7x3x4xi16>, tensor<7x4xi16>
  }
  func.func private @expected() -> tensor<7x3xi16> {
    %0 = stablehlo.constant dense<[[6, 5, -1], [-14, -15, -16], [-2, 5, 7], [-19, -4, 32], [22, 19, 13], [12, -5, -6], [-16, -2, 0]]> : tensor<7x3xi16>
    return %0 : tensor<7x3xi16>
  }
}

