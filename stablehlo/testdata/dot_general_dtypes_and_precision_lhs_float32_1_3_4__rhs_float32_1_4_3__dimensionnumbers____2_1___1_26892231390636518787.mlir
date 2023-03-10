// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>)
    %1 = call @expected() : () -> tensor<1xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>) {
    %0 = stablehlo.constant dense<[[[3.29589868, 1.23571205, 1.77729869, -1.18627894], [-0.334112585, -0.701673209, 1.37458372, 1.33988047], [-1.23167276, -1.47567272, -5.089730e+00, 4.67841911]]]> : tensor<1x3x4xf32>
    %1 = stablehlo.constant dense<[[[-1.98014688, 6.42148352, -4.60602903], [0.598645091, -1.56104231, 1.48972511], [2.90995979, -0.593875766, -4.25129318], [-1.8781029, 1.97643542, 2.91169143]]]> : tensor<1x4x3xf32>
    return %0, %1 : tensor<1x3x4xf32>, tensor<1x4x3xf32>
  }
  func.func private @expected() -> tensor<1xf32> {
    %0 = stablehlo.constant dense<41.1297302> : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
