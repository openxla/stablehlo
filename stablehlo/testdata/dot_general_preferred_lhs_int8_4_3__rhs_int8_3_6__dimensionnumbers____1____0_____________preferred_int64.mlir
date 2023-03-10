// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xi8>)
    %1 = call @expected() : () -> tensor<4x6xi32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xi8>, tensor<3x6xi8>) -> tensor<4x6xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x6xi32>, tensor<4x6xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xi8>, tensor<3x6xi8>) {
    %0 = stablehlo.constant dense<[[-1, 0, 3], [-3, -2, 0], [-3, -2, 3], [5, 3, -2]]> : tensor<4x3xi8>
    %1 = stablehlo.constant dense<[[-3, 0, -2, 0, 5, 0], [-2, -1, 0, 0, 0, 1], [-3, -3, 4, 2, 0, -1]]> : tensor<3x6xi8>
    return %0, %1 : tensor<4x3xi8>, tensor<3x6xi8>
  }
  func.func private @expected() -> tensor<4x6xi32> {
    %0 = stablehlo.constant dense<[[-6, -9, 14, 6, -5, -3], [13, 2, 6, 0, -15, -2], [4, -7, 18, 6, -15, -5], [-15, 3, -18, -4, 25, 5]]> : tensor<4x6xi32>
    return %0 : tensor<4x6xi32>
  }
}

