// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xui8>, tensor<1x4x3xui8>)
    %1 = call @expected() : () -> tensor<1xui8>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xui8>, tensor<1x4x3xui8>) -> tensor<1xui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xui8>, tensor<1xui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xui8>, tensor<1x4x3xui8>) {
    %0 = stablehlo.constant dense<[[[1, 6, 1, 0], [3, 3, 2, 3], [3, 5, 1, 4]]]> : tensor<1x3x4xui8>
    %1 = stablehlo.constant dense<[[[4, 1, 0], [4, 0, 0], [1, 1, 0], [3, 0, 1]]]> : tensor<1x4x3xui8>
    return %0, %1 : tensor<1x3x4xui8>, tensor<1x4x3xui8>
  }
  func.func private @expected() -> tensor<1xui8> {
    %0 = stablehlo.constant dense<38> : tensor<1xui8>
    return %0 : tensor<1xui8>
  }
}

