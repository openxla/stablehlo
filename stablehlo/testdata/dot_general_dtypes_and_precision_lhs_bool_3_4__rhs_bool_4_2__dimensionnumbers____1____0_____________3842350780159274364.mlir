// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xi1>, tensor<4x2xi1>)
    %1 = call @expected() : () -> tensor<3x2xi1>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xi1>, tensor<4x2xi1>) -> tensor<3x2xi1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xi1>, tensor<3x2xi1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xi1>, tensor<4x2xi1>) {
    %0 = stablehlo.constant dense<true> : tensor<3x4xi1>
    %1 = stablehlo.constant dense<true> : tensor<4x2xi1>
    return %0, %1 : tensor<3x4xi1>, tensor<4x2xi1>
  }
  func.func private @expected() -> tensor<3x2xi1> {
    %0 = stablehlo.constant dense<true> : tensor<3x2xi1>
    return %0 : tensor<3x2xi1>
  }
}

