// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xi32>)
    %1 = call @expected() : () -> tensor<4x6xi32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xi32>, tensor<3x6xi32>) -> tensor<4x6xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x6xi32>, tensor<4x6xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xi32>, tensor<3x6xi32>) {
    %0 = stablehlo.constant dense<[[0, -1, 2], [1, -3, -5], [0, -1, 2], [-2, 4, 3]]> : tensor<4x3xi32>
    %1 = stablehlo.constant dense<[[3, -2, -1, 4, 1, 2], [-4, 0, 2, 1, 2, 1], [-2, 0, 3, -1, -2, 5]]> : tensor<3x6xi32>
    return %0, %1 : tensor<4x3xi32>, tensor<3x6xi32>
  }
  func.func private @expected() -> tensor<4x6xi32> {
    %0 = stablehlo.constant dense<[[0, 0, 4, -3, -6, 9], [25, -2, -22, 6, 5, -26], [0, 0, 4, -3, -6, 9], [-28, 4, 19, -7, 0, 15]]> : tensor<4x6xi32>
    return %0 : tensor<4x6xi32>
  }
}

