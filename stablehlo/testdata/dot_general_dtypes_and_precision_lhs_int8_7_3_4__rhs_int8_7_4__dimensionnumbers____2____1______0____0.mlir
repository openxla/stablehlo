// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<7x3x4xi8>, tensor<7x4xi8>)
    %1 = call @expected() : () -> tensor<7x3xi8>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<7x3x4xi8>, tensor<7x4xi8>) -> tensor<7x3xi8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<7x3xi8>, tensor<7x3xi8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<7x3x4xi8>, tensor<7x4xi8>) {
    %0 = stablehlo.constant dense<[[[0, 1, 4, 1], [-2, -3, 0, 0], [6, -1, 0, 0]], [[1, 0, -2, 0], [1, 3, 4, -6], [2, 4, 4, 0]], [[0, -2, -1, 1], [-2, -3, 0, 2], [-3, 0, 0, -2]], [[4, -7, 2, 2], [0, 4, 2, 0], [-6, 1, 1, 2]], [[-2, -2, 0, -1], [-4, -1, 0, -1], [1, 3, 1, 1]], [[-4, 0, 0, 1], [-1, 0, 4, -2], [0, 5, 0, -1]], [[0, 2, 1, 2], [-1, 1, -3, -2], [-6, -3, -1, -3]]]> : tensor<7x3x4xi8>
    %1 = stablehlo.constant dense<[[2, 0, -1, 4], [-4, 0, 2, -1], [0, 6, 8, 0], [-1, -3, -1, -1], [-3, 0, 5, 0], [-3, 0, 3, -1], [2, 1, -2, -3]]> : tensor<7x4xi8>
    return %0, %1 : tensor<7x3x4xi8>, tensor<7x4xi8>
  }
  func.func private @expected() -> tensor<7x3xi8> {
    %0 = stablehlo.constant dense<[[0, -4, 12], [-8, 10, 0], [-20, -18, 0], [13, -14, 0], [6, 12, 2], [11, 17, 1], [-6, 11, -4]]> : tensor<7x3xi8>
    return %0 : tensor<7x3xi8>
  }
}

