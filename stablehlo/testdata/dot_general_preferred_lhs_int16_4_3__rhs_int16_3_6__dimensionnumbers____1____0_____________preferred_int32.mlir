// RUN-DISABLED: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xi16>, tensor<3x6xi16>)
    %1 = call @expected() : () -> tensor<4x6xi32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xi16>, tensor<3x6xi16>) -> tensor<4x6xi32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x6xi32>, tensor<4x6xi32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xi16>, tensor<3x6xi16>) {
    %0 = stablehlo.constant dense<[[-1, -2, 0], [1, 0, -2], [0, -3, 2], [0, -2, 0]]> : tensor<4x3xi16>
    %1 = stablehlo.constant dense<[[-3, -2, -1, 0, 0, 0], [3, 2, 2, -1, 0, 5], [1, -2, -2, 1, 1, -4]]> : tensor<3x6xi16>
    return %0, %1 : tensor<4x3xi16>, tensor<3x6xi16>
  }
  func.func private @expected() -> tensor<4x6xi32> {
    %0 = stablehlo.constant dense<[[-3, -2, -3, 2, 0, -10], [-5, 2, 3, -2, -2, 8], [-7, -10, -10, 5, 2, -23], [-6, -4, -4, 2, 0, -10]]> : tensor<4x6xi32>
    return %0 : tensor<4x6xi32>
  }
}

