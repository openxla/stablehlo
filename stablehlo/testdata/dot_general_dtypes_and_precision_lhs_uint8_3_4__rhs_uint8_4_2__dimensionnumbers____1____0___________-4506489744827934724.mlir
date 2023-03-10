// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xui8>, tensor<4x2xui8>)
    %1 = call @expected() : () -> tensor<3x2xui8>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x4xui8>, tensor<4x2xui8>) -> tensor<3x2xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xui8>, tensor<3x2xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xui8>, tensor<4x2xui8>) {
    %0 = stablehlo.constant dense<[[0, 3, 1, 1], [3, 4, 0, 2], [1, 5, 3, 0]]> : tensor<3x4xui8>
    %1 = stablehlo.constant dense<[[2, 2], [1, 0], [4, 1], [1, 0]]> : tensor<4x2xui8>
    return %0, %1 : tensor<3x4xui8>, tensor<4x2xui8>
  }
  func.func private @expected() -> tensor<3x2xui8> {
    %0 = stablehlo.constant dense<[[8, 1], [12, 6], [19, 5]]> : tensor<3x2xui8>
    return %0 : tensor<3x2xui8>
  }
}
