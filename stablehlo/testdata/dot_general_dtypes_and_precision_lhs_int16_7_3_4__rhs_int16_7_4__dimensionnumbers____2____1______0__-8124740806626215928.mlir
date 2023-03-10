// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xi16>, tensor<7x4xi16>)
    %1 = call @expected() : () -> tensor<7x3xi16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<7x3x4xi16>, tensor<7x4xi16>) -> tensor<7x3xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xi16>, tensor<7x3xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xi16>, tensor<7x4xi16>) {
    %0 = stablehlo.constant dense<[[[-6, 1, -3, 3], [-3, 0, 0, -4], [3, -1, -1, 2]], [[1, -4, 0, 0], [1, 3, 3, 4], [-1, 2, 0, 1]], [[3, 3, -2, 0], [0, 5, 1, 4], [0, 0, -2, 1]], [[-3, -2, 3, 0], [2, 5, -3, -1], [-1, 0, -4, 0]], [[0, 2, -9, -2], [0, 4, -1, 5], [0, 0, -1, 2]], [[5, -1, -3, 2], [-1, 7, -3, 1], [-1, -1, 1, 1]], [[0, 7, -5, -2], [-2, 4, 0, 2], [7, 0, 3, 0]]]> : tensor<7x3x4xi16>
    %1 = stablehlo.constant dense<[[-4, 0, 2, 0], [0, -1, 2, 2], [0, 1, 1, -3], [0, 1, -1, 8], [0, -1, 2, 0], [2, 0, 0, 1], [-1, -2, -4, 5]]> : tensor<7x4xi16>
    return %0, %1 : tensor<7x3x4xi16>, tensor<7x4xi16>
  }
  func.func private @expected() -> tensor<7x3xi16> {
    %0 = stablehlo.constant dense<[[18, 12, -14], [4, 11, 0], [1, -6, -5], [-5, 0, 4], [-20, -6, -2], [12, -1, -1], [-4, 4, -19]]> : tensor<7x3xi16>
    return %0 : tensor<7x3xi16>
  }
}
