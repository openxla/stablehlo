// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xui16>, tensor<4x2xui16>)
    %1 = call @expected() : () -> tensor<3x2xui16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x4xui16>, tensor<4x2xui16>) -> tensor<3x2xui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xui16>, tensor<3x2xui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xui16>, tensor<4x2xui16>) {
    %0 = stablehlo.constant dense<[[1, 0, 6, 6], [0, 1, 0, 0], [0, 0, 2, 4]]> : tensor<3x4xui16>
    %1 = stablehlo.constant dense<[[3, 7], [1, 1], [3, 5], [2, 2]]> : tensor<4x2xui16>
    return %0, %1 : tensor<3x4xui16>, tensor<4x2xui16>
  }
  func.func private @expected() -> tensor<3x2xui16> {
    %0 = stablehlo.constant dense<[[33, 49], [1, 1], [14, 18]]> : tensor<3x2xui16>
    return %0 : tensor<3x2xui16>
  }
}
