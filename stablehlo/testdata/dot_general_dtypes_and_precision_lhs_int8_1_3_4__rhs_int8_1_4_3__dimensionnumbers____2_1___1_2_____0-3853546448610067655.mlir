// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xi8>, tensor<1x4x3xi8>)
    %1 = call @expected() : () -> tensor<1xi8>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xi8>, tensor<1x4x3xi8>) -> tensor<1xi8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xi8>, tensor<1xi8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xi8>, tensor<1x4x3xi8>) {
    %0 = stablehlo.constant dense<[[[0, -2, 1, 1], [-2, -1, 3, 3], [-2, 0, 2, -1]]]> : tensor<1x3x4xi8>
    %1 = stablehlo.constant dense<[[[-1, 2, -1], [-1, 1, 0], [1, 0, 1], [-5, -3, -3]]]> : tensor<1x4x3xi8>
    return %0, %1 : tensor<1x3x4xi8>, tensor<1x4x3xi8>
  }
  func.func private @expected() -> tensor<1xi8> {
    %0 = stablehlo.constant dense<-9> : tensor<1xi8>
    return %0 : tensor<1xi8>
  }
}

