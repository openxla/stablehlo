// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xi8>, tensor<3x6xi8>)
    %1 = call @expected() : () -> tensor<6xi32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xi8>, tensor<3x6xi8>) -> tensor<6xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6xi32>, tensor<6xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xi8>, tensor<3x6xi8>) {
    %0 = stablehlo.constant dense<[-5, -1, -4]> : tensor<3xi8>
    %1 = stablehlo.constant dense<[[-3, 0, 0, -3, -3, -3], [3, 3, -5, 0, 0, 1], [1, -3, -2, 2, -5, 5]]> : tensor<3x6xi8>
    return %0, %1 : tensor<3xi8>, tensor<3x6xi8>
  }
  func.func private @expected() -> tensor<6xi32> {
    %0 = stablehlo.constant dense<[8, 9, 13, 7, 35, -6]> : tensor<6xi32>
    return %0 : tensor<6xi32>
  }
}

