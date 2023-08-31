// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xi16>, tensor<4x2xi16>)
    %1 = call @expected() : () -> tensor<3x2xi16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x4xi16>, tensor<4x2xi16>) -> tensor<3x2xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xi16>, tensor<3x2xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xi16>, tensor<4x2xi16>) {
    %0 = stablehlo.constant dense<[[7, 6, 1, 4], [1, 3, 4, 1], [3, -2, -1, -3]]> : tensor<3x4xi16>
    %1 = stablehlo.constant dense<[[1, 4], [-2, 2], [-1, -4], [0, 0]]> : tensor<4x2xi16>
    return %0, %1 : tensor<3x4xi16>, tensor<4x2xi16>
  }
  func.func private @expected() -> tensor<3x2xi16> {
    %0 = stablehlo.constant dense<[[-6, 36], [-9, -6], [8, 12]]> : tensor<3x2xi16>
    return %0 : tensor<3x2xi16>
  }
}
